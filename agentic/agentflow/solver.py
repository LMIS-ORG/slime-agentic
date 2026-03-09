import json
import logging
import uuid
from pathlib import Path
from typing import Any

from llm_engine import GenerationOutput
from planner import Planner
from executor import Executor
from verifier import Verifier
from formatters import extract_context_subgoal_and_tool
from memory import Memory

logger = logging.getLogger(__name__)


class Solver:
    """
    engine_map 示例：
        {
            "default":  sglang_engine,   # 必填，作为未指定模块的 fallback
            "executor": qwen_engine,
            "verifier": qwen_engine,
        }
    planner / final_output 未指定时自动使用 "default"。
    """

    MAX_TOTAL_TOKENS = 16384

    def __init__(
        self,
        engine_map: dict[str, Any],
        tools_dir: str = None,
        max_steps: int = 5,
        trajectory_dir: str = None,
    ):
        def _get(module: str):
            return engine_map.get(module) or engine_map["default"]

        self._engine = _get("default")   # 保留 default engine 用于 tokenizer 访问
        self._tools_dir = tools_dir
        self.max_steps = max_steps
        self._trajectory_dir = Path(trajectory_dir) if trajectory_dir else None
        if self._trajectory_dir:
            self._trajectory_dir.mkdir(parents=True, exist_ok=True)

        self.planner  = Planner(llm_engine=_get("planner"),  tools_dir=tools_dir)
        _module_keys = {"default", "planner", "executor", "verifier"}
        tools_engine_map = {k: v for k, v in engine_map.items() if k not in _module_keys}
        self.executor = Executor(
            llm_engine=_get("executor"),
            available_tools=self.planner.available_tools,
            tools_engine_map=tools_engine_map,
        )
        self.verifier = Verifier(llm_engine=_get("verifier"), available_tools=self.planner.available_tools, tools_metadata=self.planner.toolbox_metadata)

    def _save_trajectory(self, trajectory: dict) -> None:
        if self._trajectory_dir is None:
            return
        file_path = self._trajectory_dir / f"{uuid.uuid4().hex}.json"
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(trajectory, f, ensure_ascii=False, indent=2)
        logger.debug("trajectory saved to %s", file_path)

    async def solve(self, question: str, label: str | None = None) -> GenerationOutput:
        memory = Memory()
        trajectory = {
            "question": question,
            "label": label,
            "analysis": "",
            "steps": [],
            "final_output": "",
        }

        # 第一轮：planner analysis，模型生成参与 loss（mask=1）
        analysis = await self.planner.plan(question)
        token_ids   = list(analysis.token_ids)
        loss_mask   = [1] * len(analysis.token_ids)
        log_probs   = list(analysis.log_probs)
        response_text = analysis.response
        finish_reason = analysis.finish_reason
        trajectory["analysis"] = analysis.response

        for step_count in range(self.max_steps):
            try:
                # ── next_step prompt（注入，mask=0）+ next_step 生成（mask=1）──
                next_step_prompt_text = self.planner.get_next_step_prompt_text(
                    question, analysis.response, memory,
                    step_count=step_count, max_step_count=self.max_steps,
                )
                next_step_prompt_ids = self._engine.tokenizer(
                    next_step_prompt_text, add_special_tokens=False
                )["input_ids"]
                next_step = await self.planner.generate_next_step(
                    question, memory, analysis.response,
                    step_count=step_count, max_step_count=self.max_steps,
                )
            except Exception as e:
                logger.warning("[step %d] planner.generate_next_step failed: %s", step_count, e)
                break

            token_ids   += next_step_prompt_ids + next_step.token_ids
            loss_mask   += [0] * len(next_step_prompt_ids) + [1] * len(next_step.token_ids)
            log_probs   += [0.0] * len(next_step_prompt_ids) + next_step.log_probs
            response_text += next_step_prompt_text + next_step.response
            finish_reason = next_step.finish_reason

            context, sub_goal, tool_name = extract_context_subgoal_and_tool(next_step.response)

            # ── 执行工具（不拼入序列），结果写入 memory ──
            try:
                tool_command, _ = await self.executor.generate_tool_command(
                    question, context, sub_goal, tool_name,
                    self.planner.toolbox_metadata, step_count=step_count,
                )
            except Exception as e:
                logger.warning("[step %d] executor.generate_tool_command failed: %s", step_count, e)
                tool_command = f"Error generating command: {e}"

            execution_result = await self.executor.execute_command(tool_name, tool_command, self._tools_dir)
            memory.add_action(step_count, tool_name, sub_goal, tool_command, execution_result)
            logger.debug("[step %d] execution_result: %s", step_count, execution_result)

            # ── verifier（不拼入序列），仅用于判断是否继续 ──
            try:
                _, conclusion, _ = await self.verifier.verificate_context(
                    question, analysis.response, step_count=step_count, memory=memory,
                )
            except Exception as e:
                logger.warning("[step %d] verifier failed: %s", step_count, e)
                conclusion = "STOP"

            logger.debug("[step %d] conclusion: %s", step_count, conclusion)

            trajectory["steps"].append({
                "step_count": step_count,
                "next_step": next_step.response,
                "tool_name": tool_name,
                "sub_goal": sub_goal,
                "tool_command": tool_command,
                "execution_result": execution_result,
                "conclusion": conclusion,
            })

            if conclusion == "STOP":
                break

            if len(token_ids) >= self.MAX_TOTAL_TOKENS:
                logger.info("token budget exhausted (%d >= %d), stopping early at step %d",
                            len(token_ids), self.MAX_TOTAL_TOKENS, step_count)
                break

        # ── final_output 拼入序列（mask=0）──
        try:
            final_output = await self.planner.generate_final_output(question, memory)
            token_ids   += final_output.prompt_token_ids + final_output.token_ids
            loss_mask   += [0] * len(final_output.prompt_token_ids) + [0] * len(final_output.token_ids)
            log_probs   += [0.0] * (len(final_output.prompt_token_ids) + len(final_output.token_ids))
            response_text += final_output.prompt_text + final_output.response
            trajectory["final_output"] = final_output.response
        except Exception as e:
            logger.warning("planner.generate_final_output failed: %s", e)
            trajectory["final_output"] = ""
        trajectory["full_response"] = response_text

        self._save_trajectory(trajectory)

        return GenerationOutput(
            prompt_text=analysis.prompt_text,
            prompt_token_ids=analysis.prompt_token_ids,
            response=response_text,
            token_ids=token_ids,
            log_probs=log_probs,
            finish_reason=finish_reason,
            loss_mask=loss_mask,
            final_output=final_output.response,
        )

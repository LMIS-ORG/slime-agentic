import importlib.util
import json
import logging
import re
import sys
from pathlib import Path
from typing import List, Tuple

from llm_engine import SGLangEngine, GenerationOutput
from formatters import QueryAnalysis
from memory import Memory

logger = logging.getLogger(__name__)


class Planner:
    def __init__(self, llm_engine: SGLangEngine, available_tools: List = None, toolbox_metadata: dict = None,
                 tools_dir: str = None):
        self.llm_engine = llm_engine

        if available_tools is None and toolbox_metadata is None and tools_dir is not None:
            available_tools, toolbox_metadata = self._discover_tools(tools_dir)

        self.available_tools = available_tools or []
        self.toolbox_metadata = toolbox_metadata or {}

    @staticmethod
    def _preload_tools_base(tools_path: Path):
        """将 tools/base.py 以固定名称 'tools.base' 注册到 sys.modules，
        使各 tool.py 中 `from tools.base import ...` 能够直接命中缓存，
        不依赖 sys.path 的顺序或环境差异。"""
        import types as _types

        base_file = tools_path / "base.py"
        if not base_file.exists() or "tools.base" in sys.modules:
            return

        # 注册虚拟父包 'tools'（如果尚未存在）
        if "tools" not in sys.modules:
            tools_pkg = _types.ModuleType("tools")
            tools_pkg.__path__ = [str(tools_path)]
            tools_pkg.__package__ = "tools"
            sys.modules["tools"] = tools_pkg

        spec = importlib.util.spec_from_file_location("tools.base", base_file)
        base_mod = importlib.util.module_from_spec(spec)
        sys.modules["tools.base"] = base_mod
        spec.loader.exec_module(base_mod)
        sys.modules["tools"].base = base_mod

    @staticmethod
    def _discover_tools(tools_dir: str):
        """扫描 tools 目录，加载每个子目录下 tool.py 中的 TOOL_NAME 和 TOOL_DESCRIPTION。"""
        available_tools = []
        toolbox_metadata = {}

        tools_path = Path(tools_dir)
        # 预先把 tools.base 注册进 sys.modules，让 tool.py 的导入稳定命中
        Planner._preload_tools_base(tools_path)

        for tool_file in sorted(tools_path.rglob("tool.py")):
            module_name = f"_tool_{tool_file.parent.name}"
            spec = importlib.util.spec_from_file_location(module_name, tool_file)
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            try:
                spec.loader.exec_module(module)
            except Exception as e:
                logger.warning("[Planner] 跳过 %s，加载失败: %s", tool_file, e)
                continue

            tool_name = getattr(module, "TOOL_NAME", None)
            tool_description = getattr(module, "TOOL_DESCRIPTION", None)

            if tool_name:
                available_tools.append(tool_name)
                toolbox_metadata[tool_name] = {
                    "description": tool_description or "",
                }
                #print(f"[Planner] 已加载工具: {tool_name}")

        return available_tools, toolbox_metadata

    def _build_prompt(self, question: str) -> str:
        return f"""Task: Analyze the given query to determine necessary skills and tools.

Inputs:
- Query: {question}
- Available tools: {self.available_tools}
- Metadata for tools: {self.toolbox_metadata}

Instructions:
1. Identify the main objectives in the query.
2. List the necessary skills and tools.
3. For each skill and tool, explain how it helps address the query.
4. Note any additional considerations.

"""


    async def plan(self, question: str) -> GenerationOutput:
        planning_prompt = self._build_prompt(question)
        messages = [{"role": "user", "content": planning_prompt}]
        return await self.llm_engine.generate(messages)

    def _build_next_step_content(
        self,
        query: str,
        query_analysis: str,
        memory: Memory,
        step_count: int,
        max_step_count: int,
    ) -> str:
        """构建 next_step 轮次的 user 消息内容（供 generate_next_step 与 get_next_step_prompt_text 复用）。"""
        return f"""
Task: Determine the optimal next step to address the query using available tools and previous steps.
Context:
- **Query:** {query}
- **Query Analysis:** {query_analysis}
- **Available Tools:** {self.available_tools}
- **Toolbox Metadata:** {self.toolbox_metadata}
- **Previous Steps:** {memory.get_actions()}

Instructions:
1. Analyze the query, previous steps, and available tools.
2. Select the **single best tool** for the next step.
3. Formulate a specific, achievable **sub-goal** for that tool.
4. Provide all necessary **context** (data, file names, variables) for the tool to function.

Response Format:
1.  **Justification:** Explain your choice of tool and sub-goal.
2.  **Context:** Provide all necessary information for the tool.
3.  **Sub-Goal:** State the specific objective for the tool.
4.  **Tool Name:** State the exact name of the selected tool.

Rules:
- Select only ONE tool.
- The sub-goal must be directly achievable by the selected tool.
- The Context section must contain all information the tool needs to function.
- The response must end with the Context, Sub-Goal, and Tool Name sections in that order, with no extra content.

"""

    def get_next_step_prompt_text(
        self,
        query: str,
        query_analysis: str,
        memory: Memory,
        step_count: int,
        max_step_count: int,
    ) -> str:
        """返回 next_step 轮次发送给模型的完整 prompt 文本（含 chat template），用于 tokenize 以构造 loss_mask。"""
        content = self._build_next_step_content(query, query_analysis, memory, step_count, max_step_count)
        messages = [{"role": "user", "content": content}]
        return self.llm_engine.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            #enable_thinking=False,
        )



    async def generate_next_step(
        self,
        query: str,
        memory: Memory,
        query_analysis: str,
        step_count: int,
        max_step_count: int,
    ) -> GenerationOutput:
        content = self._build_next_step_content(query, query_analysis, memory, step_count, max_step_count)
        messages = [{"role": "user", "content": content}]
        return await self.llm_engine.generate(messages)

    async def generate_final_output(self, question: str, memory: Memory) -> str:
        prompt_generate_final_output = f"""
Task: Generate the final output based on the query and the results from all tools used.

Context:
- **Query:** {question}
- **Actions Taken:** {memory.get_actions()}

Instructions:
1. Review the query and the results from all tool executions.
2. Incorporate the relevant information to create a coherent, step-by-step final output.
3. You MUST end your response with the final answer enclosed in \\boxed{{}}. For example: \\boxed{{42}}.
"""
        messages = [{"role": "user", "content": prompt_generate_final_output}]
        return await self.llm_engine.generate(messages)
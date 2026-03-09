#!/usr/bin/env python3
"""
AgentFlow 独立评估脚本
仅前向推理，不涉及任何训练逻辑，直接复用 Solver / Rewarder 流水线。

典型用法（服务器已在运行）：
    python eval_agentflow.py \\
        --tokenizer /data/model/qwen25_7b/ \\
        --eval-data aime /data/aime-2024/aime-2024.jsonl \\
        --output eval_results.json \\
        --concurrency 16

典型用法（自动拉起 SGLang 服务器）：
    python eval_agentflow.py \\
        --model /data/AgentFlow_pro-Qwen25-7B-RL/ \\
        --start-servers --tp 4 \\
        --eval-data aime /data/aime-2024/aime-2024.jsonl \\
        --output eval_results.json
"""

import argparse
import asyncio
import json
import logging
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import httpx

# ── PYTHONPATH 保证在任意目录下也能 import agentflow 模块 ───────────────────────
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from llm_engine import SGLangEngine          # noqa: E402
from solver import Solver                    # noqa: E402
from rewarder import Rewarder                # noqa: E402
from slime.rollout.rm_hub.math_dapo_utils import (   # noqa: E402
    last_boxed_only_string,
    normalize_final_answer,
    remove_boxed,
)
import slime.utils.http_utils as _http_utils  # noqa: E402


def _init_http_client(concurrency: int = 256) -> None:
    """初始化 slime http_utils 的全局 AsyncClient。
    训练流程通过 init_http_client(args) 完成此步骤；eval 脚本绕过了训练框架，
    需要在发起任何 SGLang 请求前手动调用一次。
    """
    if _http_utils._http_client is None:
        _http_utils._http_client = httpx.AsyncClient(
            limits=httpx.Limits(max_connections=concurrency),
            timeout=httpx.Timeout(None),
        )

# ──────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("eval_agentflow")

TOOLS_DIR = _SCRIPT_DIR / "tools"


# ── 数据加载 ──────────────────────────────────────────────────────────────────

def load_dataset(path: str, input_key: str = "prompt", label_key: str = "label") -> list[dict]:
    """从 JSONL 文件加载 {question, label} 列表。支持 chat messages 格式的 input_key。"""
    samples = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            question = obj.get(input_key, "")
            label = obj.get(label_key, "")
            # chat messages 格式 → 取最后一条 user 消息
            if isinstance(question, list):
                for msg in reversed(question):
                    if isinstance(msg, dict) and msg.get("role") == "user":
                        question = msg["content"]
                        break
                else:
                    question = str(question)
            samples.append({"question": str(question), "label": str(label)})
    return samples


# ── 答案提取工具 ───────────────────────────────────────────────────────────────

def _extract_pred(text: str) -> str:
    """从回复中提取预测答案（优先 \\boxed{...}，否则返回末行截断）。"""
    boxed = last_boxed_only_string(text)
    if boxed:
        return normalize_final_answer(remove_boxed(boxed))
    lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
    return lines[-1][:200] if lines else ""


# ── SGLang 服务器管理 ──────────────────────────────────────────────────────────

def _wait_for_server(url: str, timeout: int = 300, interval: int = 5) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            urllib.request.urlopen(url, timeout=3)
            return True
        except Exception:
            time.sleep(interval)
    return False


class SGLangServer:
    """在子进程中启动并管理一个 SGLang HTTP 服务。"""

    def __init__(
        self,
        model_path: str,
        port: int,
        tp: int,
        mem_fraction: float,
        ctx_len: int,
        extra_args: list[str] | None = None,
    ):
        self.port = port
        cmd = [
            sys.executable, "-m", "sglang.launch_server",
            "--model-path", model_path,
            "--port", str(port),
            "--tp", str(tp),
            "--mem-fraction-static", str(mem_fraction),
            "--context-length", str(ctx_len),
            "--trust-remote-code",
        ] + (extra_args or [])
        logger.info("启动 SGLang 服务 (port=%d): %s", port, " ".join(cmd))
        self._proc = subprocess.Popen(cmd)

    def wait_ready(self, timeout: int = 300) -> bool:
        url = f"http://127.0.0.1:{self.port}/health"
        logger.info("等待 port=%d 就绪（最多 %ds）…", self.port, timeout)
        ok = _wait_for_server(url, timeout=timeout)
        if ok:
            logger.info("port=%d 已就绪。", self.port)
        else:
            logger.error("port=%d 在 %ds 内未就绪。", self.port, timeout)
        return ok

    def stop(self):
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self._proc.kill()


# ── 单条样本评估 ───────────────────────────────────────────────────────────────

async def _eval_one(
    solver: Solver,
    rewarder: Rewarder,
    question: str,
    label: str,
    sampling_params: dict,
    semaphore: asyncio.Semaphore,
    idx: int,
    total: int,
) -> dict:
    async with semaphore:
        try:
            out = await solver.solve(question, label=label)
        except Exception as exc:
            logger.warning("[%d/%d] Solver 异常: %s", idx + 1, total, exc)
            return {
                "idx": idx,
                "question": question,
                "label": label,
                "pred": "",
                "final_output": "",
                "score": 0.0,
                "error": str(exc),
            }

        if out is None:
            return {
                "idx": idx,
                "question": question,
                "label": label,
                "pred": "",
                "final_output": "",
                "score": 0.0,
                "error": "solver returned None",
            }

        final_output = out.final_output or ""
        pred = _extract_pred(final_output)

        if pred and label and pred == label:
            score = 1.0
        else:
            try:
                score = await rewarder.compute_reward(
                    question=question,
                    model_response=final_output,
                    groundtruth=label,
                )
            except Exception as exc:
                logger.warning("[%d/%d] Rewarder 异常: %s", idx + 1, total, exc)
                score = 0.0

        logger.info(
            "[%d/%d] score=%.1f | pred=%.40s | label=%.40s",
            idx + 1, total, score, pred, label,
        )
        return {
            "idx": idx,
            "question": question,
            "label": label,
            "pred": pred,
            "final_output": final_output,
            "score": score,
        }


# ── 批量评估主协程 ─────────────────────────────────────────────────────────────

async def run_eval(
    samples: list[dict],
    planner_url: str,
    executor_url: str,
    coder_url: str,
    tokenizer,
    sampling_params: dict,
    concurrency: int,
    max_steps: int,
    trajectory_dir: str | None,
) -> list[dict]:
    """并发地对所有 samples 跑 Solver + Rewarder，返回结果列表。

    三个引擎与 rollout.py 中保持一致：
      planner_engine  → "planner" / "default"
                        对应 rollout.py 的 engine（sglang_router）
      executor_engine → "executor" / "verifier" / "base_generator" / rewarder
                        对应 rollout.py 的 generate_engine（硬编码 port 30000）
      coder_engine    → "python_coder"
                        对应 rollout.py 的 coder_engine（硬编码 port 30001）
    """
    # 确保 slime http_utils 的全局 HTTP 客户端已初始化（训练路径由框架完成，eval 需手动触发）
    _init_http_client(concurrency=concurrency * 4)

    max_new_tokens = sampling_params.get("max_new_tokens", 2048)

    # Planner：主模型，负责 plan / next_step / final_output，不开 thinking
    planner_engine = SGLangEngine(
        url=planner_url,
        tokenizer=tokenizer,
        sampling_params=sampling_params,
        max_new_tokens=max_new_tokens,
        enable_thinking=False,
    )
    # Executor / base_generator：通用生成引擎
    executor_engine = SGLangEngine(
        url=executor_url,
        tokenizer=tokenizer,
        sampling_params=sampling_params,
        max_new_tokens=max_new_tokens,
    )
    # Coder / python_coder：代码生成引擎
    coder_engine = SGLangEngine(
        url=coder_url,
        tokenizer=tokenizer,
        sampling_params=sampling_params,
        max_new_tokens=max_new_tokens,
    )
    # Rewarder 与 reward_func 保持一致：使用 generate_engine（对应 executor_url）
    rewarder_engine = SGLangEngine(
        url=executor_url,
        tokenizer=tokenizer,
        sampling_params={},
        max_new_tokens=512,
    )

    # verifier 与 rollout.py 保持一致：使用 executor_engine（generate_engine，非 coder_engine）
    engine_map = {
        "default":        planner_engine,
        "planner":        planner_engine,
        "executor":       executor_engine,
        "verifier":       executor_engine,
        "base_generator": executor_engine,
        "python_coder":   coder_engine,
    }

    solver = Solver(
        engine_map=engine_map,
        tools_dir=str(TOOLS_DIR),
        trajectory_dir=trajectory_dir,
        max_steps=max_steps,
    )
    rewarder = Rewarder(llm_engine=rewarder_engine)
    semaphore = asyncio.Semaphore(concurrency)
    total = len(samples)

    tasks = [
        _eval_one(
            solver, rewarder,
            s["question"], s["label"],
            sampling_params, semaphore,
            i, total,
        )
        for i, s in enumerate(samples)
    ]
    results = await asyncio.gather(*tasks)
    return sorted(results, key=lambda r: r["idx"])  # 保持原始顺序


# ── CLI 参数解析 ───────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="AgentFlow 独立评估（仅前向推理）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # 模型 / Tokenizer
    model_grp = p.add_argument_group("模型配置")
    model_grp.add_argument("--model", default=None,
                           help="HF 模型路径（--start-servers 时必填）")
    model_grp.add_argument("--tokenizer", default=None,
                           help="HF tokenizer 路径（默认同 --model）")

    # 服务器连接（服务已在运行时使用）
    # 对应 rollout.py 中的三个引擎：
    #   planner_url   → engine          (sglang_router / 主模型)
    #   executor_url  → generate_engine (port 30000，executor / base_generator)
    #   coder_url     → coder_engine    (port 30001，verifier / python_coder)
    srv_grp = p.add_argument_group("服务器连接")
    srv_grp.add_argument("--planner-url",  default="http://127.0.0.1:30000/generate",
                         help="Planner / default 引擎的 SGLang generate URL")
    srv_grp.add_argument("--executor-url", default="http://127.0.0.1:30001/generate",
                         help="Executor / base_generator 引擎的 SGLang generate URL")
    srv_grp.add_argument("--coder-url",    default="http://127.0.0.1:30002/generate",
                         help="Verifier / python_coder 引擎的 SGLang generate URL")

    # 自动拉起服务器
    auto_grp = p.add_argument_group("自动拉起 SGLang 服务器")
    auto_grp.add_argument("--start-servers", action="store_true",
                          help="自动拉起三个 SGLang 服务器（需要 --model）")
    auto_grp.add_argument("--planner-port",  type=int, default=30000,
                          help="Planner 服务器端口")
    auto_grp.add_argument("--executor-port", type=int, default=30001,
                          help="Executor 服务器端口")
    auto_grp.add_argument("--coder-port",    type=int, default=30002,
                          help="Coder 服务器端口")
    auto_grp.add_argument("--tp",         type=int, default=4,
                          help="每个服务器的 Tensor Parallel 大小")
    auto_grp.add_argument("--mem-fraction", type=float, default=0.7,
                          help="SGLang mem-fraction-static")
    auto_grp.add_argument("--ctx-len",    type=int, default=65536,
                          help="SGLang context length")

    # 评估数据：--eval-data NAME PATH [NAME2 PATH2 ...]
    data_grp = p.add_argument_group("评估数据")
    data_grp.add_argument("--eval-data", nargs="+", metavar="NAME_OR_PATH",
                          help="数据集列表，格式：名称 路径 [名称 路径 ...]")
    data_grp.add_argument("--input-key", default="prompt", help="问题字段名")
    data_grp.add_argument("--label-key", default="label",  help="答案字段名")
    data_grp.add_argument("--num-samples", type=int, default=None,
                          help="每个数据集最多取多少条（调试用）")

    # 采样参数
    samp_grp = p.add_argument_group("采样参数")
    samp_grp.add_argument("--temperature",    type=float, default=0.0)
    samp_grp.add_argument("--top-p",          type=float, default=0.95)
    samp_grp.add_argument("--max-new-tokens", type=int,   default=2048)

    # 推理控制
    infer_grp = p.add_argument_group("推理控制")
    infer_grp.add_argument("--concurrency", type=int, default=16,
                           help="并发评估的最大协程数")
    infer_grp.add_argument("--max-steps",   type=int, default=5,
                           help="Solver 最大工具调用步数")

    # 输出
    out_grp = p.add_argument_group("输出")
    out_grp.add_argument("--output",         default="eval_results.json",
                         help="保存完整评估结果的 JSON 文件路径")
    out_grp.add_argument("--trajectory-dir", default=None,
                         help="若设置，将 Solver 轨迹保存到此目录")
    out_grp.add_argument("--verbose", action="store_true",
                         help="开启 DEBUG 日志")

    return p.parse_args()


# ── 入口 ───────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # ── Tokenizer ─────────────────────────────────────────────────────────────
    tokenizer_path = args.tokenizer or args.model
    if not tokenizer_path:
        logger.error("请提供 --tokenizer 或 --model 参数。")
        sys.exit(1)
    from transformers import AutoTokenizer
    logger.info("加载 tokenizer：%s", tokenizer_path)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)

    # ── 数据集 ─────────────────────────────────────────────────────────────────
    if not args.eval_data or len(args.eval_data) % 2 != 0:
        logger.error("--eval-data 需要以「名称 路径」为一组的偶数个参数，例如：\n"
                     "  --eval-data aime /data/aime-2024/aime-2024.jsonl")
        sys.exit(1)

    datasets: dict[str, list[dict]] = {}
    it = iter(args.eval_data)
    for name, path in zip(it, it):
        logger.info("加载数据集 '%s'：%s", name, path)
        samples = load_dataset(path, args.input_key, args.label_key)
        if args.num_samples:
            samples = samples[:args.num_samples]
        datasets[name] = samples
        logger.info("  → %d 条样本", len(samples))

    # ── 服务器 ─────────────────────────────────────────────────────────────────
    # 三个引擎 URL 对应关系（与 rollout.py 保持一致）：
    #   planner_url  → sglang_router / 主模型 (planner / default)
    #   executor_url → generate_engine        (executor / base_generator)
    #   coder_url    → coder_engine            (verifier / python_coder)
    servers: list[SGLangServer] = []
    planner_url  = args.planner_url
    executor_url = args.executor_url
    coder_url    = args.coder_url

    if args.start_servers:
        if not args.model:
            logger.error("--start-servers 需要同时指定 --model。")
            sys.exit(1)

        planner_srv = SGLangServer(
            args.model, args.planner_port, args.tp,
            args.mem_fraction, args.ctx_len,
        )
        executor_srv = SGLangServer(
            args.model, args.executor_port, args.tp,
            args.mem_fraction, args.ctx_len,
        )
        coder_srv = SGLangServer(
            args.model, args.coder_port, args.tp,
            args.mem_fraction, args.ctx_len,
        )
        servers.extend([planner_srv, executor_srv, coder_srv])

        planner_url  = f"http://127.0.0.1:{args.planner_port}/generate"
        executor_url = f"http://127.0.0.1:{args.executor_port}/generate"
        coder_url    = f"http://127.0.0.1:{args.coder_port}/generate"

        for srv in servers:
            if not srv.wait_ready(timeout=300):
                logger.error("SGLang 服务器启动失败，中止评估。")
                for s in servers:
                    s.stop()
                sys.exit(1)

    sampling_params = {
        "temperature":    args.temperature,
        "top_p":          args.top_p,
        "max_new_tokens": args.max_new_tokens,
    }

    # ── 评估循环 ───────────────────────────────────────────────────────────────
    all_results: dict[str, dict] = {}
    try:
        for dataset_name, samples in datasets.items():
            logger.info(
                "开始评估 '%s'（%d 条，concurrency=%d，max_steps=%d）…",
                dataset_name, len(samples), args.concurrency, args.max_steps,
            )
            t0 = time.time()

            results = asyncio.run(
                run_eval(
                    samples=samples,
                    planner_url=planner_url,
                    executor_url=executor_url,
                    coder_url=coder_url,
                    tokenizer=tokenizer,
                    sampling_params=sampling_params,
                    concurrency=args.concurrency,
                    max_steps=args.max_steps,
                    trajectory_dir=args.trajectory_dir,
                )
            )

            elapsed = time.time() - t0
            scores = [r["score"] for r in results]
            accuracy = sum(scores) / len(scores) if scores else 0.0

            logger.info(
                "数据集 '%s'：accuracy=%.3f（%d/%d）耗时 %.1fs",
                dataset_name, accuracy, int(sum(scores)), len(scores), elapsed,
            )

            all_results[dataset_name] = {
                "accuracy":       accuracy,
                "num_correct":    int(sum(scores)),
                "num_total":      len(scores),
                "elapsed_seconds": round(elapsed, 2),
                "details":        results,
            }

    finally:
        for srv in servers:
            srv.stop()

    # ── 保存结果 ───────────────────────────────────────────────────────────────
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(all_results, indent=2, ensure_ascii=False))
    logger.info("结果已保存至：%s", output_path)

    # ── 打印汇总 ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("评估结果汇总")
    print("=" * 60)
    for name, res in all_results.items():
        print(
            f"  {name:20s}  {res['accuracy']:.1%}"
            f"  ({res['num_correct']}/{res['num_total']})"
            f"  {res['elapsed_seconds']:.1f}s"
        )
    print("=" * 60)


if __name__ == "__main__":
    main()

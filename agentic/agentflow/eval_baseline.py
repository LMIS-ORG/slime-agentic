#!/usr/bin/env python3
"""
Baseline 评估脚本
不使用 AgentFlow 框架，直接单轮 QA：问题 → 模型回答 → 判断正确性 → 汇总得分。

用法（服务器已在运行）：
    python eval_baseline.py \\
        --tokenizer /data/model/qwen25_7b/ \\
        --eval-data aime /data/aime-2024/aime-2024.jsonl \\
        --output baseline_results.json

用法（自动拉起 SGLang 服务器）：
    python eval_baseline.py \\
        --model /data/AgentFlow_pro-Qwen25-7B-RL/ \\
        --start-server --tp 4 \\
        --eval-data aime /data/aime-2024/aime-2024.jsonl \\
        --output baseline_results.json
"""

import argparse
import asyncio
import json
import logging
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import httpx

# ── 保证 agentflow 目录可 import ────────────────────────────────────────────────
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from llm_engine import SGLangEngine                          # noqa: E402
from rewarder import Rewarder                                # noqa: E402
from slime.rollout.rm_hub.math_dapo_utils import (          # noqa: E402
    last_boxed_only_string,
    normalize_final_answer,
    remove_boxed,
)
import slime.utils.http_utils as _http_utils                # noqa: E402

# ──────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("eval_baseline")


# ── HTTP 客户端初始化 ──────────────────────────────────────────────────────────

def _init_http_client(concurrency: int = 256) -> None:
    """初始化 slime http_utils 的全局 AsyncClient（训练框架会自动做，eval 需手动触发）。"""
    if _http_utils._http_client is None:
        _http_utils._http_client = httpx.AsyncClient(
            limits=httpx.Limits(max_connections=concurrency),
            timeout=httpx.Timeout(None),
        )


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
    def __init__(self, model_path: str, port: int, tp: int,
                 mem_fraction: float, ctx_len: int):
        self.port = port
        cmd = [
            sys.executable, "-m", "sglang.launch_server",
            "--model-path", model_path,
            "--port", str(port),
            "--tp", str(tp),
            "--mem-fraction-static", str(mem_fraction),
            "--context-length", str(ctx_len),
            "--trust-remote-code",
        ]
        logger.info("启动 SGLang 服务 (port=%d)", port)
        self._proc = subprocess.Popen(cmd)

    def wait_ready(self, timeout: int = 300) -> bool:
        url = f"http://127.0.0.1:{self.port}/health"
        logger.info("等待 port=%d 就绪（最多 %ds）…", self.port, timeout)
        ok = _wait_for_server(url, timeout=timeout)
        if ok:
            logger.info("port=%d 已就绪。", self.port)
        else:
            logger.error("port=%d 启动超时。", self.port)
        return ok

    def stop(self):
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self._proc.kill()


# ── 数据加载 ──────────────────────────────────────────────────────────────────

def load_dataset(path: str, input_key: str = "prompt",
                 label_key: str = "label") -> list[dict]:
    """从 JSONL 文件加载 {question, label} 列表，支持 chat messages 格式。"""
    samples = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            question = obj.get(input_key, "")
            label = obj.get(label_key, "")
            if isinstance(question, list):
                for msg in reversed(question):
                    if isinstance(msg, dict) and msg.get("role") == "user":
                        question = msg["content"]
                        break
                else:
                    question = str(question)
            samples.append({"question": str(question), "label": str(label)})
    return samples


# ── 答案提取 ───────────────────────────────────────────────────────────────────

def _extract_pred(text: str) -> str:
    """优先提取 \\boxed{...}，否则返回末行截断。"""
    boxed = last_boxed_only_string(text)
    if boxed:
        return normalize_final_answer(remove_boxed(boxed))
    lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
    return lines[-1][:200] if lines else ""


# ── 单条样本评估 ───────────────────────────────────────────────────────────────

async def _eval_one(
    engine: SGLangEngine,
    rewarder: Rewarder,
    tokenizer,
    question: str,
    label: str,
    sampling_params: dict,
    system_prompt: str | None,
    semaphore: asyncio.Semaphore,
    idx: int,
    total: int,
) -> dict:
    async with semaphore:
        # 构造消息
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": question})

        try:
            out = await engine.generate(messages, sampling_params=sampling_params)
            response = out.response
        except Exception as exc:
            logger.warning("[%d/%d] generate 异常: %s", idx + 1, total, exc)
            return {
                "idx": idx, "question": question, "label": label,
                "pred": "", "response": "", "score": 0.0, "error": str(exc),
            }

        pred = _extract_pred(response)

        # 快速路径：pred 与 label 字符串完全匹配直接给 1 分
        if pred and label and pred == label:
            score = 1.0
        else:
            try:
                score = await rewarder.compute_reward(
                    question=question,
                    model_response=response,
                    groundtruth=label,
                )
            except Exception as exc:
                logger.warning("[%d/%d] rewarder 异常: %s", idx + 1, total, exc)
                score = 0.0

        logger.info(
            "[%d/%d] score=%.1f | pred=%.40s | label=%.40s",
            idx + 1, total, score, pred, label,
        )
        return {
            "idx":      idx,
            "question": question,
            "label":    label,
            "pred":     pred,
            "response": response,
            "score":    score,
        }


# ── 批量评估主协程 ─────────────────────────────────────────────────────────────

async def run_eval(
    samples: list[dict],
    model_url: str,
    rewarder_url: str,
    tokenizer,
    sampling_params: dict,
    system_prompt: str | None,
    concurrency: int,
) -> list[dict]:
    _init_http_client(concurrency=concurrency * 4)

    engine = SGLangEngine(
        url=model_url,
        tokenizer=tokenizer,
        sampling_params=sampling_params,
        max_new_tokens=sampling_params.get("max_new_tokens", 32768),
        enable_thinking=False,
    )
    rewarder_engine = SGLangEngine(
        url=rewarder_url,
        tokenizer=tokenizer,
        sampling_params={},
        max_new_tokens=512,
    )
    rewarder = Rewarder(llm_engine=rewarder_engine)
    semaphore = asyncio.Semaphore(concurrency)
    total = len(samples)

    tasks = [
        _eval_one(
            engine, rewarder, tokenizer,
            s["question"], s["label"],
            sampling_params, system_prompt,
            semaphore, i, total,
        )
        for i, s in enumerate(samples)
    ]
    results = await asyncio.gather(*tasks)
    return sorted(results, key=lambda r: r["idx"])


# ── CLI 参数解析 ───────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Baseline 评估（单轮 QA，不使用 AgentFlow）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # 模型 / tokenizer
    g = p.add_argument_group("模型配置")
    g.add_argument("--model",     default=None, help="HF 模型路径（--start-server 时必填）")
    g.add_argument("--tokenizer", default=None, help="HF tokenizer 路径（默认同 --model）")

    # 服务器连接
    g = p.add_argument_group("服务器连接（服务已在运行时使用）")
    g.add_argument("--model-url",    default="http://127.0.0.1:30000/generate",
                   help="模型推理 SGLang URL")
    g.add_argument("--rewarder-url", default="http://127.0.0.1:30000/generate",
                   help="Rewarder 使用的 SGLang URL（默认与模型共用同一服务器）")

    # 自动拉起服务器
    g = p.add_argument_group("自动拉起 SGLang 服务器")
    g.add_argument("--start-server", action="store_true",
                   help="自动拉起 SGLang 服务器（需要 --model）")
    g.add_argument("--port",         type=int,   default=30000)
    g.add_argument("--tp",           type=int,   default=4)
    g.add_argument("--mem-fraction", type=float, default=0.7)
    g.add_argument("--ctx-len",      type=int,   default=65536)

    # 评估数据
    g = p.add_argument_group("评估数据")
    g.add_argument("--eval-data",   nargs="+", metavar="NAME_OR_PATH",
                   help="数据集列表，格式：名称 路径 [名称 路径 ...]")
    g.add_argument("--input-key",   default="prompt")
    g.add_argument("--label-key",   default="label")
    g.add_argument("--num-samples", type=int, default=None,
                   help="每个数据集最多取多少条（调试用）")

    # 采样参数
    g = p.add_argument_group("采样参数")
    g.add_argument("--temperature",    type=float, default=0.0)
    g.add_argument("--top-p",          type=float, default=0.95)
    g.add_argument("--max-new-tokens", type=int,   default=4096)
    g.add_argument("--system-prompt",  type=str,   default=None,
                   help="可选 system prompt；不填则无 system 消息")

    # 推理控制
    g = p.add_argument_group("推理控制")
    g.add_argument("--concurrency", type=int, default=32,
                   help="并发评估的最大协程数")

    # 输出
    g = p.add_argument_group("输出")
    g.add_argument("--output",  default="baseline_results.json")
    g.add_argument("--verbose", action="store_true")

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
            samples = samples[: args.num_samples]
        datasets[name] = samples
        logger.info("  → %d 条样本", len(samples))

    # ── 服务器 ─────────────────────────────────────────────────────────────────
    server = None
    model_url    = args.model_url
    rewarder_url = args.rewarder_url

    if args.start_server:
        if not args.model:
            logger.error("--start-server 需要同时指定 --model。")
            sys.exit(1)
        server = SGLangServer(
            args.model, args.port, args.tp,
            args.mem_fraction, args.ctx_len,
        )
        if not server.wait_ready(timeout=300):
            logger.error("SGLang 服务器启动失败，中止评估。")
            server.stop()
            sys.exit(1)
        model_url    = f"http://127.0.0.1:{args.port}/generate"
        rewarder_url = model_url

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
                "开始评估 '%s'（%d 条，concurrency=%d）…",
                dataset_name, len(samples), args.concurrency,
            )
            t0 = time.time()

            results = asyncio.run(
                run_eval(
                    samples=samples,
                    model_url=model_url,
                    rewarder_url=rewarder_url,
                    tokenizer=tokenizer,
                    sampling_params=sampling_params,
                    system_prompt=args.system_prompt,
                    concurrency=args.concurrency,
                )
            )

            elapsed = time.time() - t0
            scores  = [r["score"] for r in results]
            accuracy = sum(scores) / len(scores) if scores else 0.0

            logger.info(
                "数据集 '%s'：accuracy=%.3f（%d/%d）耗时 %.1fs",
                dataset_name, accuracy, int(sum(scores)), len(scores), elapsed,
            )

            all_results[dataset_name] = {
                "accuracy":        accuracy,
                "num_correct":     int(sum(scores)),
                "num_total":       len(scores),
                "elapsed_seconds": round(elapsed, 2),
                "details":         results,
            }

    finally:
        if server:
            server.stop()

    # ── 保存结果 ───────────────────────────────────────────────────────────────
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(all_results, indent=2, ensure_ascii=False))
    logger.info("结果已保存至：%s", output_path)

    # ── 打印汇总 ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Baseline 评估结果汇总")
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

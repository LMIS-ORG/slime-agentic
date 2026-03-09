import asyncio
import os
from dataclasses import dataclass, field
from typing import Any

from slime.utils.http_utils import post


@dataclass
class GenerationOutput:
    prompt_text: str
    prompt_token_ids: list[int]
    response: str
    token_ids: list[int]
    log_probs: list[float]
    finish_reason: str  # "stop" | "length" | "abort"
    # 多轮时由 solver 填充：与 token_ids 等长，1=模型生成参与 loss，0=注入的 prompt 不参与
    loss_mask: list[int] | None = None
    # planner 的最终综合回答，用于 reward 打分
    final_output: str | None = None


class SGLangEngine:
    """
    对 SGLang HTTP /generate 接口的轻量封装。
    各模块（Planner、Solver 等）持有同一个实例，统一发起 LLM 调用。
    """

    def __init__(self, url: str, tokenizer: Any, sampling_params: dict, max_new_tokens: int | None = None, enable_thinking: bool = False):
        self.url = url
        self.tokenizer = tokenizer
        self.sampling_params = dict(sampling_params)
        self.enable_thinking = enable_thinking
        if max_new_tokens is not None:
            self.sampling_params["max_new_tokens"] = max_new_tokens

    async def generate(
        self,
        messages: list[dict[str, str]],
        sampling_params: dict | None = None,
    ) -> GenerationOutput:
        """
        接受标准 chat messages，返回 GenerationOutput。
        sampling_params 若传入则覆盖初始化时的默认值。
        """
        params = sampling_params if sampling_params is not None else self.sampling_params

        prompt_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=self.enable_thinking,
        )
        prompt_token_ids = self.tokenizer(prompt_text, add_special_tokens=False)["input_ids"]

        payload = {
            "text": prompt_text,
            "sampling_params": params,
            "return_logprob": True,
        }
        out = await post(self.url, payload)

        meta = out["meta_info"]
        finish_reason = meta["finish_reason"]["type"]
        response = out["text"]
        token_ids = [item[1] for item in meta["output_token_logprobs"]]
        log_probs  = [item[0] for item in meta["output_token_logprobs"]]

        return GenerationOutput(
            prompt_text=prompt_text,
            prompt_token_ids=prompt_token_ids,
            response=response,
            token_ids=token_ids,
            log_probs=log_probs,
            finish_reason=finish_reason,
        )


_qwen_semaphore: asyncio.Semaphore | None = None


def _get_qwen_semaphore(concurrency: int) -> asyncio.Semaphore:
    global _qwen_semaphore
    if _qwen_semaphore is None:
        _qwen_semaphore = asyncio.Semaphore(concurrency)
    return _qwen_semaphore


class QwenEngine:
    """
    通过 DashScope OpenAI 兼容接口调用 Qwen 模型。
    接口与 SGLangEngine 保持一致，可无感替换。
    所有实例共享同一个全局信号量，避免并发请求打爆 DashScope 限流。
    """

    def __init__(
        self,
        model: str = "qwen-flash",
        api_key: str | None = None,
        base_url: str = "http://127.0.0.1:30000/v1",
        max_tokens: int = 4096,
        concurrency: int = 128,
        **default_params,
    ):
        from openai import AsyncOpenAI

        self.model = model
        self.max_tokens = max_tokens
        self.default_params = default_params
        self.concurrency = concurrency
        # 本地 sglang 服务不需要真实 api_key，用 "EMPTY" 作占位；
        # 若指向 DashScope 等远程服务，则优先取传入的 api_key 或环境变量。
        resolved_api_key = api_key or os.getenv("DASHSCOPE_API_KEY") or "EMPTY"
        self.client = AsyncOpenAI(
            api_key=resolved_api_key,
            base_url=base_url,
            timeout=120.0,
        )

    async def generate(
        self,
        messages: list[dict[str, str]],
        sampling_params: dict | None = None,
        max_retries: int = 8,
    ) -> GenerationOutput:
        """
        接受标准 chat messages，返回 GenerationOutput。
        sampling_params 支持 max_tokens / temperature / top_p 等标准字段。
        超时、连接失败、限流（503）自动指数退避重试。
        """
        import openai

        params = {**self.default_params, **(sampling_params or {})}
        max_tokens = params.pop("max_new_tokens", None) or params.pop("max_tokens", None) or self.max_tokens

        semaphore = _get_qwen_semaphore(self.concurrency)
        for attempt in range(max_retries):
            try:
                async with semaphore:
                    completion = await self.client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        max_tokens=max_tokens,
                        **params,
                    )
                break
            except (
                openai.APITimeoutError,
                openai.APIConnectionError,
                openai.InternalServerError,
                openai.RateLimitError,
            ) as e:
                if attempt + 1 >= max_retries:
                    logger.error(f"QwenEngine.generate failed after {max_retries} attempts: {e}")
                    raise
                backoff = min(2 ** attempt, 60) + random.random()
                logger.warning(
                    f"QwenEngine retryable error ({type(e).__name__}), "
                    f"retry {attempt + 1}/{max_retries} in {backoff:.1f}s"
                )
                await asyncio.sleep(backoff)

        choice = completion.choices[0]
        response = choice.message.content or ""
        finish_reason = choice.finish_reason or "stop"

        prompt_text = "\n".join(f"{m['role']}: {m['content']}" for m in messages)

        return GenerationOutput(
            prompt_text=prompt_text,
            prompt_token_ids=[],
            response=response,
            token_ids=[],
            log_probs=[],
            finish_reason=finish_reason,
        )

#!/bin/bash
# Baseline 评估脚本
# 单轮 QA，不使用 AgentFlow 框架，只需一个 SGLang 服务器。
#
# 用法：
#   bash eval_baseline.sh                        # 使用默认参数
#   NUM_SAMPLES=5 bash eval_baseline.sh          # 仅跑 5 条（快速调试）
#   AUTO_START=0 bash eval_baseline.sh           # 手动模式（服务器已在运行）

set -e

# ── 可按需修改的配置 ──────────────────────────────────────────────────────────

# 待评估的模型权重（HF 格式）
MODEL_PATH=${MODEL_PATH:-"/data/model/qwen25_7b/"}

# Tokenizer 路径
TOKENIZER_PATH=${TOKENIZER_PATH:-"/data/model/qwen25_7b/"}

# 评估数据集（格式：名称 JSONL路径，可追加多组）
EVAL_DATA=(
    aime /data/aime-2024/aime-2024.jsonl
)

# 输出文件路径
OUTPUT=${OUTPUT:-"$(dirname "$0")/baseline_results.json"}

# SGLang 服务器端口（baseline 只需一个）
PORT=${PORT:-30000}

# Tensor Parallel 大小
TP=${TP:-4}

# SGLang 显存占用比
MEM_FRACTION=${MEM_FRACTION:-0.7}

# SGLang context length
CTX_LEN=${CTX_LEN:-65536}

# 并发评估协程数（baseline 无工具调用，可以设大一些）
CONCURRENCY=${CONCURRENCY:-32}

# 采样参数
TEMPERATURE=${TEMPERATURE:-0.0}
TOP_P=${TOP_P:-0.95}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-4096}

# 调试：限制样本数（0 = 不限制）
NUM_SAMPLES=${NUM_SAMPLES:-0}

# 是否自动拉起 SGLang 服务器（设为 1 则自动拉起）
AUTO_START=${AUTO_START:-1}

# ── 环境准备 ──────────────────────────────────────────────────────────────────

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

export PYTHONPATH="/root/Megatron-LM/:${SCRIPT_DIR}:/root/slime:${PYTHONPATH:-}"

# ── 构建 Python 命令参数 ───────────────────────────────────────────────────────

PY_ARGS=(
    --tokenizer      "${TOKENIZER_PATH}"
    --eval-data      "${EVAL_DATA[@]}"
    --input-key      prompt
    --label-key      label
    --output         "${OUTPUT}"
    --concurrency    "${CONCURRENCY}"
    --temperature    "${TEMPERATURE}"
    --top-p          "${TOP_P}"
    --max-new-tokens "${MAX_NEW_TOKENS}"
    --port           "${PORT}"
    --tp             "${TP}"
    --mem-fraction   "${MEM_FRACTION}"
    --ctx-len        "${CTX_LEN}"
)

if [ "${AUTO_START}" = "1" ]; then
    PY_ARGS+=(--model "${MODEL_PATH}" --start-server)
else
    PY_ARGS+=(--model-url "http://127.0.0.1:${PORT}/generate")
fi

if [ "${NUM_SAMPLES}" -gt 0 ] 2>/dev/null; then
    PY_ARGS+=(--num-samples "${NUM_SAMPLES}")
fi

# ── 如果手动模式，提示用户先启动服务器 ────────────────────────────────────────

if [ "${AUTO_START}" != "1" ]; then
    echo "============================================================"
    echo " 手动模式：请确保 SGLang 服务器已在运行："
    echo "   模型服务器 (推理 + rewarder) : port ${PORT}"
    echo ""
    echo " 快速启动示例："
    echo "   python -m sglang.launch_server \\"
    echo "     --model-path ${MODEL_PATH} --port ${PORT} \\"
    echo "     --tp ${TP} --mem-fraction-static ${MEM_FRACTION} \\"
    echo "     --context-length ${CTX_LEN} --trust-remote-code &"
    echo "============================================================"
    echo ""
fi

# ── 运行评估 ───────────────────────────────────────────────────────────────────

echo "▶ 开始 Baseline 评估..."
echo "  模型       : ${MODEL_PATH}"
echo "  Tokenizer  : ${TOKENIZER_PATH}"
echo "  输出文件   : ${OUTPUT}"
echo "  并发数     : ${CONCURRENCY}"
echo ""

python3 "${SCRIPT_DIR}/eval_baseline.py" "${PY_ARGS[@]}"

echo ""
echo "✓ 评估完成，结果保存至：${OUTPUT}"

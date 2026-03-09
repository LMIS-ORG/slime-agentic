#!/bin/bash
# AgentFlow 独立评估脚本
# 先手动拉起 SGLang 服务器，再调用 eval_agentflow.py 进行纯推理评估。
#
# 用法：
#   bash eval_agentflow.sh                        # 使用默认参数
#   NUM_SAMPLES=5 bash eval_agentflow.sh          # 仅跑 5 条（快速调试）
#   CONCURRENCY=32 bash eval_agentflow.sh         # 调大并发

set -e

# ── 可按需修改的配置 ──────────────────────────────────────────────────────────

# 训练完成后的模型权重（HF 格式）
MODEL_PATH=${MODEL_PATH:-"/data/AgentFlow_pro-Qwen25-7B-RL/"}

# Tokenizer 路径（通常与原始 HF 基座模型保持一致）
TOKENIZER_PATH=${TOKENIZER_PATH:-"/data/model/qwen25_7b/"}

# 评估数据集（格式：名称 JSONL路径，可追加多组）
EVAL_DATA=(
    aime /data/aime-2024/aime-2024.jsonl
)

# 输出文件路径
OUTPUT=${OUTPUT:-"$(dirname "$0")/eval_results.json"}

# 轨迹保存目录（设为空字符串则不保存）
TRAJECTORY_DIR=${TRAJECTORY_DIR:-""}

# Tensor Parallel 大小（显卡数 ÷ 服务器数）
TP=${TP:-4}

# SGLang 显存占用比
MEM_FRACTION=${MEM_FRACTION:-0.7}

# SGLang context length
CTX_LEN=${CTX_LEN:-65536}

# 并发评估协程数（越大越快，但受显存和服务器吞吐限制）
CONCURRENCY=${CONCURRENCY:-16}

# 每条问题的最大工具调用步数
MAX_STEPS=${MAX_STEPS:-5}

# 采样参数（temperature=0 为贪心，推理时常用）
TEMPERATURE=${TEMPERATURE:-0.0}
TOP_P=${TOP_P:-0.95}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-2048}

# 调试：限制样本数（0 = 不限制）
NUM_SAMPLES=${NUM_SAMPLES:-0}

# SGLang 三个服务器端口（对应 rollout.py 中的三个引擎）
#   PLANNER_PORT  → Planner / default  (主模型)
#   EXECUTOR_PORT → Executor / base_generator
#   CODER_PORT    → Verifier / python_coder
PLANNER_PORT=${PLANNER_PORT:-30000}
EXECUTOR_PORT=${EXECUTOR_PORT:-30001}
CODER_PORT=${CODER_PORT:-30002}

# 是否让脚本自动拉起 SGLang（设为 1 则自动拉起，否则需要手动提前启动）
AUTO_START=${AUTO_START:-1}

# ── 环境准备 ──────────────────────────────────────────────────────────────────

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

export PYTHONPATH="/root/Megatron-LM/:${SCRIPT_DIR}:/root/slime:${PYTHONPATH:-}"

# ── 构建 Python 命令参数 ───────────────────────────────────────────────────────

PY_ARGS=(
    --tokenizer  "${TOKENIZER_PATH}"
    --eval-data  "${EVAL_DATA[@]}"
    --input-key  prompt
    --label-key  label
    --output     "${OUTPUT}"
    --concurrency "${CONCURRENCY}"
    --max-steps  "${MAX_STEPS}"
    --temperature "${TEMPERATURE}"
    --top-p       "${TOP_P}"
    --max-new-tokens "${MAX_NEW_TOKENS}"
    --tp          "${TP}"
    --mem-fraction "${MEM_FRACTION}"
    --ctx-len     "${CTX_LEN}"
    --planner-port  "${PLANNER_PORT}"
    --executor-port "${EXECUTOR_PORT}"
    --coder-port    "${CODER_PORT}"
)

if [ "${AUTO_START}" = "1" ]; then
    PY_ARGS+=(--model "${MODEL_PATH}" --start-servers)
else
    # 手动模式：指向已运行的服务器 URL
    PY_ARGS+=(
        --planner-url  "http://127.0.0.1:${PLANNER_PORT}/generate"
        --executor-url "http://127.0.0.1:${EXECUTOR_PORT}/generate"
        --coder-url    "http://127.0.0.1:${CODER_PORT}/generate"
    )
fi

if [ -n "${TRAJECTORY_DIR}" ]; then
    PY_ARGS+=(--trajectory-dir "${TRAJECTORY_DIR}")
fi

if [ "${NUM_SAMPLES}" -gt 0 ] 2>/dev/null; then
    PY_ARGS+=(--num-samples "${NUM_SAMPLES}")
fi

# ── 如果手动模式，提示用户先启动服务器 ────────────────────────────────────────

if [ "${AUTO_START}" != "1" ]; then
    echo "============================================================"
    echo " 手动模式：请确保以下三个 SGLang 服务器已在运行："
    echo "   Planner  服务器 (planner / default)                    : port ${PLANNER_PORT}"
    echo "   Executor 服务器 (executor / verifier / base_generator) : port ${EXECUTOR_PORT}"
    echo "   Coder    服务器 (python_coder)                         : port ${CODER_PORT}"
    echo ""
    echo " 快速启动示例（三个服务器分别使用不同的模型）："
    echo "   python -m sglang.launch_server \\"
    echo "     --model-path ${MODEL_PATH} --port ${PLANNER_PORT} \\"
    echo "     --tp ${TP} --mem-fraction-static ${MEM_FRACTION} \\"
    echo "     --context-length ${CTX_LEN} --trust-remote-code &"
    echo ""
    echo "   python -m sglang.launch_server \\"
    echo "     --model-path Qwen/Qwen2.5-7B --port ${EXECUTOR_PORT} \\"
    echo "     --tp ${TP} --mem-fraction-static ${MEM_FRACTION} \\"
    echo "     --context-length ${CTX_LEN} --trust-remote-code &"
    echo ""
    echo "   python -m sglang.launch_server \\"
    echo "     --model-path Qwen/Qwen2.5-Coder-7B-Instruct --port ${CODER_PORT} \\"
    echo "     --tp ${TP} --mem-fraction-static ${MEM_FRACTION} \\"
    echo "     --context-length ${CTX_LEN} --trust-remote-code &"
    echo "============================================================"
    echo ""
fi

# ── 运行评估 ───────────────────────────────────────────────────────────────────

echo "▶ 开始评估..."
echo "  模型       : ${MODEL_PATH}"
echo "  Tokenizer  : ${TOKENIZER_PATH}"
echo "  输出文件   : ${OUTPUT}"
echo "  并发数     : ${CONCURRENCY}"
echo "  最大步数   : ${MAX_STEPS}"
echo ""

python3 "${SCRIPT_DIR}/eval_agentflow.py" "${PY_ARGS[@]}"

echo ""
echo "✓ 评估完成，结果保存至：${OUTPUT}"

#!/bin/bash


# Cleanup previous runs
pkill -9 sglang
sleep 3
ray stop --force
pkill -9 ray
pkill -9 python
sleep 3
pkill -9 ray
pkill -9 python

set -ex

# Set to "1" to save training trajectories to trajectories/ folder (default: off)
SAVE_TRAJECTORY=${SAVE_TRAJECTORY:-"0"}

# Reset trajectories directory (only when saving is enabled)
TRAJECTORIES_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)/trajectories"
if [ "${SAVE_TRAJECTORY:-0}" = "1" ]; then
    rm -rf "${TRAJECTORIES_DIR}"
    mkdir -p "${TRAJECTORIES_DIR}"
fi

# Prevent ray from buffering stdout/stderr
export PYTHONBUFFERED=16

# Detect NVLink availability
NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then
    HAS_NVLINK=1
else
    HAS_NVLINK=0
fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

# Get script directory
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

# Load model configuration
source "${SCRIPT_DIR}/../../scripts/models/qwen2.5-7B.sh"

# Checkpoint arguments
CKPT_ARGS=(
   --hf-checkpoint /data/Qwen2.5-7B-Instruct
   --ref-load /data/qwen2.5_7b_dist/
   --save /data/AgentFlow_Qwen25-7B-RL/
   --save-interval 100
)

# Rollout arguments
ROLLOUT_ARGS=(
   --prompt-data /data/dapo-math-17k/dapo-math-17k.jsonl
   --input-key prompt
   --label-key label
   --rollout-shuffle
   --reward-key score
   --num-epoch 1                  
   --rollout-batch-size 8     
   --n-samples-per-prompt 8       
   --rollout-max-response-len 32768 
   --rollout-temperature 0.7      
   --global-batch-size 64   
   --balance-data
)

# Evaluation arguments
EVAL_ARGS=(
   --eval-interval 20
   --eval-prompt-data aime /data/aime-2024/aime-2024.jsonl
   --n-samples-per-eval-prompt 1
   --eval-max-response-len 32768

   --eval-top-p 0.95
)

# Performance arguments
PERF_ARGS=(
   --tensor-model-parallel-size 4  # 训练侧保留 TP=4（7B 模型稳定性）
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1

   --recompute-granularity full    
   --recompute-method uniform
   --recompute-num-layers 1

   --use-dynamic-batch-size
   --max-tokens-per-gpu 16384
)

# GRPO arguments
GRPO_ARGS=(
   --advantage-estimator grpo     
   --use-kl-loss                  
   --kl-loss-coef 0.001           
   --kl-loss-type low_var_kl
   --entropy-coef 0.0              
   --eps-clip 0.2                 
   --eps-clip-high 0.3            
)

# Optimizer arguments
OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6                      
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
)

# WandB arguments
# config: trainer.logger=['console','wandb'], PROJECT_NAME='AgentFlow_pro'
# 启用 wandb：请确保环境变量 WANDB_KEY 已设置，或手动填入 key
# WANDB_ARGS=(
#    --use-wandb
#    --wandb-project AgentFlow_pro
#    --wandb-group AgentFlow_pro-Qwen25-7B-RL
#    --wandb-key ${WANDB_KEY:-"your_wandb_key_here"}
# )
# 如不需要 wandb，注释上面 4 行并取消注释下一行：
WANDB_ARGS=()

# SGLang arguments
# config: ROLLOUT_TP_SIZE=1, gpu_memory_utilization=0.6
# SGLang arguments
SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 4
   --sglang-mem-fraction-static 0.75
   --sglang-context-length 65536
)

# Misc arguments
MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
)

# Custom ReAct generation and reward functions
CUSTOM_ARGS=(
   --custom-generate-function-path rollout.generate
   --custom-rm-path rollout.reward_func
   --custom-eval-rollout-log-function-path rollout.eval_log
   --custom-convert-samples-to-train-data-path custom_convert.custom_convert
)

# Launch Ray head node
# config: N_GPUS=8 (原: 4)
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus 4 --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

# Build runtime environment JSON
RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/root/Megatron-LM/:${SCRIPT_DIR}:/root/slime\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
    \"SAVE_TRAJECTORY\": \"${SAVE_TRAJECTORY}\"
  }
}"

# Submit training job
ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 train.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node 4 \
   --colocate \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${GRPO_ARGS[@]} \
   ${WANDB_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${EVAL_ARGS[@]} \
   ${SGLANG_ARGS[@]} \
   ${MISC_ARGS[@]} \
   ${CUSTOM_ARGS[@]}

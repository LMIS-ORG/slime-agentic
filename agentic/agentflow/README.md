# AgentFlow — Agentic RL with Slime

Reproduction of AgentFlow based on the slime framework, applying GRPO to multi-step agent trajectories for RL training on mathematical reasoning tasks.

[中文文档](./README_zh.md)

---

## 1. Download Datasets

```bash
# Training dataset (dapo-math-17k)
huggingface-cli download --repo-type dataset zhuzilin/dapo-math-17k \
  --local-dir /data/dapo-math-17k

# Evaluation dataset (aime-2024)
huggingface-cli download --repo-type dataset zhuzilin/aime-2024 \
  --local-dir /data/aime-2024
```

## 2. Download Model

```bash
huggingface-cli download Qwen/Qwen2.5-7B-Instruct \
  --local-dir /data/Qwen2.5-7B-Instruct
```

## 3. Convert Model Format

slime training requires converting HuggingFace checkpoints to Megatron distributed format:
```bash
cd /path/to/slime-agentic
source scripts/models/qwen2.5-7B.sh 

python tools/convert_hf_to_torch_dist.py \
  ${MODEL_ARGS[@]} \
  --hf-checkpoint /data/Qwen2.5-7B-Instruct \
  --save /data/qwen2.5_7b_dist/
```

convert Megatron distributed format model to huggingface format:
```bash
cd /path/to/slime-agentic
source scripts/models/qwen2.5-7B.sh
PYTHONPATH=/root/Megatron-LM python tools/convert_torch_dist_to_hf.py   --input-dir /data/AgentFlow_Qwen25-7B-RL/iter_00000xxx/   --output-dir /data/agentflow_xxx_hf   --origin-hf-dir /data/Qwen2.5-7B-Instruct
```

- `--input-dir`：the path of the torch_dist checkpoint saved during training
- `--output-dir`：the path of the converted HuggingFace format checkpoint
- `--origin-hf-dir`：the path of the original HuggingFace model (used to fill in config files)


## 4. Training

### 4.1 Start SGLang Services

Two SGLang inference services must be running before training starts (one for Executor/Verifier, one for Planner):

Note: The script agentflow_qwen25_7b_rl_v2.sh will kill all Python processes. Therefore, it is recommended to start the SGLang inference service only after the training script has finished killing processes and begun loading the training process.

```bash
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1

# Coder model for Executor / Verifier (port 30001)
CUDA_VISIBLE_DEVICES=4,5 python3 -m sglang.launch_server \
  --model Qwen/Qwen2.5-Coder-7B-Instruct \
  --port 30001 \
  --context-length 65536 \
  --tp 2

# Base model for Planner (port 30000, default)
CUDA_VISIBLE_DEVICES=6,7 python3 -m sglang.launch_server \
  --model Qwen/Qwen2.5-7B-Instruct \
  --port 30000 \
  --context-length 65536 \
  --tp 2
```

 these two SGLang services are required for training

### 4.2 Launch Training

Switch to the project root directory and run:

```bash
cd /path/to/slime
bash agentic/agentflow/agentflow_qwen25_7b_rl_v2.sh
```

Checkpoints are saved to `/data/AgentFlow_Qwen25-7B-RL/` by default. This can be changed in the `CKPT_ARGS` section of the script.

To save full agent trajectories for debugging:

```bash
cd /path/to/slime
SAVE_TRAJECTORY=1 bash agentic/agentflow/agentflow_qwen25_7b_rl_v2.sh
```

Trajectories are saved as JSON files under `trajectories/`, each containing the full `analysis → steps → final_output` chain.

## 5. Key Training Parameters

| Parameter | Value | Description |
|---|---|---|
| `--advantage-estimator` | `grpo` | Advantage estimation algorithm |
| `--lr` | `1e-6` | Learning rate |
| `--n-samples-per-prompt` | `8` | Number of samples per prompt |
| `--rollout-batch-size` | `8` | Rollout batch size |
| `--rollout-temperature` | `0.7` | Sampling temperature |
| `--kl-loss-coef` | `0.001` | KL divergence coefficient |
| `--eps-clip` / `--eps-clip-high` | `0.2` / `0.3` | PPO clip range |
| `--sglang-context-length` | `65536` | SGLang context length |

## 6. Evaluation

Training automatically evaluates on AIME 2024 every 20 steps, logging results to `eval_scores.json`.

### 6.1 Convert Checkpoint to HF Format

Before manual evaluation, convert the saved torch_dist checkpoint back to HuggingFace format:

```bash
cd /path/to/slime
source scripts/models/qwen2.5-7B.sh

PYTHONPATH=/root/Megatron-LM python tools/convert_torch_dist_to_hf.py \
    ${MODEL_ARGS[@]} \
    --load /data/AgentFlow_Qwen25-7B-RL/ \
    --hf-checkpoint /data/Qwen2.5-7B-Instruct \
    --save /data/AgentFlow_Qwen25-7B-RL-hf/
```

- `--load`: Path to the torch_dist checkpoint saved during training
- `--hf-checkpoint`: Original HuggingFace model path (used to fill in config files)
- `--save`: Output path for the converted HuggingFace checkpoint

### 6.2 Start Three SGLang Services

The evaluation agent pipeline uses three independent services on separate ports:

| Port | Role | Model |
|---|---|---|
| 30000 | Planner (main model) | Trained AgentFlow model |
| 30001 | Executor / Verifier / base_generator | Base chat model |
| 30002 | python_coder | Coder model |

```bash
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1

# Planner: trained model (port 30000)
CUDA_VISIBLE_DEVICES=0,1 python3 -m sglang.launch_server \
  --model /data/AgentFlow_Qwen25-7B-RL-hf/ \
  --port 30000 \
  --context-length 65536 \
  --tp 2 &

# Executor / Verifier / base_generator (port 30001)
CUDA_VISIBLE_DEVICES=2,3 python3 -m sglang.launch_server \
  --model Qwen/Qwen2.5-7B-Instruct \
  --port 30001 \
  --context-length 65536 \
  --tp 2 &

# python_coder (port 30002)
CUDA_VISIBLE_DEVICES=4,5 python3 -m sglang.launch_server \
  --model Qwen/Qwen2.5-Coder-7B-Instruct \
  --port 30002 \
  --context-length 65536 \
  --tp 2 &
```

Wait for all three services to be ready, then run:

### 6.3 Run Evaluation

```bash
AUTO_START=0 bash agentic/agentflow/eval_agentflow.sh
```

Results are saved to `agentic/agentflow/eval_results.json`.

## 7. Baseline Evaluation

The baseline uses single-turn direct inference without the AgentFlow framework, requiring only one SGLang service. To run with a manually started service:

```bash
# Start the SGLang service (port 30000)
CUDA_VISIBLE_DEVICES=0,1 python3 -m sglang.launch_server \
  --model Qwen/Qwen2.5-7B-Instruct \
  --port 30000 \
  --context-length 65536 \
  --tp 2 &

# Run evaluation
AUTO_START=0 bash agentic/agentflow/eval_baseline.sh
```

Results are saved to `agentic/agentflow/baseline_results.json`.

## 8. Adding Custom Tools

You can extend AgentFlow with new tools by adding a subdirectory under `tools/`. The Planner auto-discovers all tools at startup, so no registration is needed.

Each tool directory must contain a `tool.py` with the following structure:

```python
from tools.base import BaseTool

# Required: tool name exposed to the Planner
TOOL_NAME = "My_Custom_Tool"

# Required: description telling the Planner when and how to use this tool
TOOL_DESCRIPTION = """
A brief description of what this tool does and when to use it.
"""

# Required: demo commands showing the Planner how to call this tool
TOOL_DEMO_COMMANDS = [
    {
        "command": 'execution = tool.execute(query="example query")',
        "description": "What this example does.",
    },
]


class My_Custom_Tool(BaseTool):
    def __init__(self, llm_engine):
        super().__init__(
            tool_name=TOOL_NAME,
            tool_description=TOOL_DESCRIPTION,
            demo_commands=TOOL_DEMO_COMMANDS,
        )
        self.llm_engine = llm_engine

    async def execute(self, query: str) -> str:
        # Unified execution entry point — all tools must implement this method.
        # query: the sub-goal string generated by the Planner.
        # Returns a string result that will be stored in Memory and fed back to the Planner.
        ...
```

Key points:
- `TOOL_NAME`, `TOOL_DESCRIPTION`, and `TOOL_DEMO_COMMANDS` are read by the Planner to decide which tool to call and how to formulate the command.
- `execute(query)` is the **unified execution entry point** called by the Executor for every tool. It receives the sub-goal string from the Planner and must return a string result.
- `llm_engine` is injected automatically if the tool needs to call an LLM internally (e.g., for code generation). Leave it out if the tool is purely deterministic.

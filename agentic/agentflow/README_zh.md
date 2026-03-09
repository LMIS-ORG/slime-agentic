# AgentFlow — Agentic RL with Slime

基于 slime 框架复现 AgentFlow，在数学推理任务上使用 GRPO 对多步 agent 轨迹进行强化学习训练。

[English](./README.md)

---

## 1. 下载数据集

```bash
# 下载训练集（dapo-math-17k）
huggingface-cli download --repo-type dataset zhuzilin/dapo-math-17k \
  --local-dir /data/dapo-math-17k

# 下载评测集（aime-2024）
huggingface-cli download --repo-type dataset zhuzilin/aime-2024 \
  --local-dir /data/aime-2024
```

## 2. 下载模型

```bash
# 下载 Qwen2.5-7B-Instruct
huggingface-cli download Qwen/Qwen2.5-7B-Instruct \
  --local-dir /data/Qwen2.5-7B-Instruct
```

## 3. 转换模型格式

slime 训练需要将 HuggingFace checkpoint 转换为 Megatron 分布式格式：

```bash
cd /path/to/slime
python tools/convert_hf_to_torch_dist.py \
  --hf-checkpoint /data/Qwen2.5-7B-Instruct \
  --save /data/qwen2.5_7b_dist/
```

## 4. 启动训练

### 4.1 启动 SGLang 服务

训练前需先启动两个 SGLang 推理服务（分别用于 Executor/Verifier 和 Planner）：

```bash
# Executor / Verifier 使用的 coder 模型（端口 30001）
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
CUDA_VISIBLE_DEVICES=4,5 python3 -m sglang.launch_server \
  --model Qwen/Qwen2.5-Coder-7B-Instruct \
  --port 30001 \
  --context-length 65536 \
  --tp 2

# Planner 使用的基础模型（端口 30000，默认端口）
CUDA_VISIBLE_DEVICES=6,7 python3 -m sglang.launch_server \
  --model Qwen/Qwen2.5-7B-Instruct \
  --context-length 65536 \
  --tp 2
```

等待两个服务均就绪后，再执行训练。

### 4.2 启动训练

切换到项目根目录，再执行以下命令启动训练：

```bash
cd /path/to/slime
bash agentic/agentflow/agentflow_qwen25_7b_rl_v2.sh
```

训练权重默认保存至 `/data/AgentFlow_-Qwen25-7B-RL/`，可在脚本中的 `CKPT_ARGS` 部分修改。

如需保存每条样本的完整 agent 轨迹（用于调试），设置环境变量：

```bash
cd /path/to/slime
SAVE_TRAJECTORY=1 bash agentic/agentflow/agentflow_qwen25_7b_rl_v2.sh
```

轨迹以 JSON 格式保存至 `trajectories/`，每条记录包含 `analysis → steps → final_output` 完整链路。

## 5. 关键训练参数

| 参数 | 值 | 说明 |
|---|---|---|
| `--advantage-estimator` | `grpo` | 优势估计算法 |
| `--lr` | `1e-6` | 学习率 |
| `--n-samples-per-prompt` | `8` | 每条 prompt 采样数 |
| `--rollout-batch-size` | `8` | rollout batch size |
| `--rollout-temperature` | `0.7` | 采样温度 |
| `--kl-loss-coef` | `0.001` | KL 散度系数 |
| `--eps-clip` / `--eps-clip-high` | `0.2` / `0.3` | PPO clip 范围 |
| `--sglang-context-length` | `65536` | SGLang 上下文长度 |

## 6. 评测

### 6.1 转换模型格式

训练过程中每 20 步在 AIME 2024 上自动评测，结果记录于 `eval_scores.json`。

手动评测前，需先将训练保存的 torch_dist 格式 checkpoint 转换回 HuggingFace 格式：

```bash
cd /path/to/slime
source scripts/models/qwen2.5-7B.sh

PYTHONPATH=/root/Megatron-LM python tools/convert_torch_dist_to_hf.py \
    ${MODEL_ARGS[@]} \
    --load /data/AgentFlow_-Qwen25-7B-RL/ \
    --hf-checkpoint /data/Qwen2.5-7B-Instruct \
    --save /data/AgentFlow_Qwen25-7B-RL-hf/
```

- `--load`：训练产出的 torch_dist checkpoint 路径
- `--hf-checkpoint`：原始 HuggingFace 模型路径（用于补全配置文件）
- `--save`：转换后的 HuggingFace 格式保存路径

转换完成后，需手动启动三个 SGLang 推理服务，再执行评测脚本。

### 6.2 启动三个 SGLang 服务

评测阶段的 agent 流程涉及三个独立角色，分别占用不同端口：

| 端口 | 角色 | 模型 |
|---|---|---|
| 30000 | Planner（主模型） | 训练后的 AgentFlow 模型 |
| 30001 | Executor / Verifier / base_generator | 基础对话模型 |
| 30002 | python_coder | Coder 模型 |

```bash
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1

# Planner：使用训练后的模型（端口 30000）
CUDA_VISIBLE_DEVICES=0,1 python3 -m sglang.launch_server \
  --model /data/AgentFlow_Qwen25-7B-RL-hf/ \
  --port 30000 \
  --context-length 65536 \
  --tp 2 &

# Executor / Verifier / base_generator（端口 30001）
CUDA_VISIBLE_DEVICES=2,3 python3 -m sglang.launch_server \
  --model Qwen/Qwen2.5-7B-Instruct \
  --port 30001 \
  --context-length 65536 \
  --tp 2 &

# python_coder（端口 30002）
CUDA_VISIBLE_DEVICES=4,5 python3 -m sglang.launch_server \
  --model Qwen/Qwen2.5-Coder-7B-Instruct \
  --port 30002 \
  --context-length 65536 \
  --tp 2 &
```

等待三个服务均就绪后，执行评测。

### 6.3 执行评测

```bash
AUTO_START=0 bash agentic/agentflow/eval_agentflow.sh
```

结果保存至 `agentic/agentflow/eval_results.json`。

## 7. 评测 Baseline

Baseline 为单轮直接推理，不使用 AgentFlow 框架，只需一个 SGLang 服务。
如需手动指定已运行的服务：

```bash
# 先启动 SGLang 服务（端口 30000）
CUDA_VISIBLE_DEVICES=0,1 python3 -m sglang.launch_server \
  --model Qwen/Qwen2.5-7B-Instruct \
  --port 30000 \
  --context-length 65536 \
  --tp 2 &

# 再执行评测
AUTO_START=0 bash agentic/agentflow/eval_baseline.sh
```

结果保存至 `agentic/agentflow/baseline_results.json`。

## 8. 添加自定义工具

在 `tools/` 目录下新建一个子目录即可扩展工具，Planner 在启动时会自动扫描并加载所有工具，无需额外注册。

每个工具目录下需包含一个 `tool.py`，结构如下：

```python
from tools.base import BaseTool

# 必填：工具名称，Planner 通过此名称识别和调用该工具
TOOL_NAME = "My_Custom_Tool"

# 必填：工具描述，告诉 Planner 这个工具的用途和使用时机
TOOL_DESCRIPTION = """
简要描述该工具的功能及适用场景。
"""

# 必填：调用示例，帮助 Planner 理解如何构造调用命令
TOOL_DEMO_COMMANDS = [
    {
        "command": 'execution = tool.execute(query="示例查询")',
        "description": "该示例的说明。",
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
        # 统一执行入口，所有工具都必须实现此方法。
        # query：Planner 生成的子目标字符串。
        # 返回字符串结果，将被写入 Memory 并反馈给 Planner。
        ...
```

注意事项：
- `TOOL_NAME`、`TOOL_DESCRIPTION`、`TOOL_DEMO_COMMANDS` 由 Planner 读取，用于决定调用哪个工具以及如何构造调用命令。
- `execute(query)` 是**所有工具的统一执行入口**，由 Executor 统一调用，接收 Planner 生成的子目标字符串，必须返回一个字符串结果。
- 如果工具内部需要调用 LLM（如代码生成），`llm_engine` 会自动注入；纯确定性工具可以不使用。

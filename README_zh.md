# Agentic RL Reproductions with Slime

基于 [slime](https://github.com/THUDM/slime) 框架，复现各类 **Agentic RL** 训练方法。

slime 提供了灵活的自定义 `generate` 函数与 reward 函数接口，使得将多步 agent rollout 接入 RL 训练流程变得简洁高效。本仓库在此基础上实现并复现若干有代表性的 Agentic RL 算法，所有实现均位于 [`agentic/`](./agentic) 目录下。

---

## 已实现方法

### AgentFlow — `agentic/agentflow/`

复现 [AgentFlow](https://arxiv.org/abs/2510.05592) 的核心思路：将单步 LLM 推理扩展为多轮 **Planner → Executor → Verifier** 的 agent 循环，并对 Planner 的生成轨迹施加 RL 信号（GRPO），从而在不依赖人工标注中间步骤的情况下提升模型的工具调用与推理能力。

#### 架构

```
问题输入
  │
  ▼
Planner.plan()          ← 分析问题，制定解题思路（loss_mask=1，参与训练）
  │
  └─► for step in range(max_steps):
        │
        ├─ Planner.generate_next_step()   ← 选择下一步工具及子目标（loss_mask=1）
        ├─ Executor.generate_tool_command() + execute_command()  ← 调用工具（不计入序列）
        ├─ Verifier.verificate_context()  ← 判断是否继续（不计入序列）
        └─ Memory.add_action()            ← 记录执行结果
  │
  ▼
Planner.generate_final_output()  ← 汇总结果，输出最终答案（loss_mask=0）
  │
  ▼
Rewarder.compute_reward()        ← LLM-as-Judge，对比模型答案与 ground truth
```

#### 工具（`tools/`）

| 工具名 | 描述 |
|---|---|
| `base_generator` | 通用文本生成工具，基于 LLM 直接回答子任务 |
| `python_coder` | Python 代码生成与执行工具，用于数学计算、算法求解等 |

#### 实验结果

| 模型 | 数据集 | Baseline | AgentFlow（复现） | 提升 |
|---|---|---|---|---|
| Qwen2.5-7B-Instruct | AIME 2024 | 10.0% | 26.7% | +16.7% |

#### 训练配置

- **算法**：GRPO，KL 散度约束（`low_var_kl`）
- **模型**：Qwen2.5-7B（可替换）
- **数据**：DAPO-Math-17K
- **推理引擎**：SGLang（Planner / Executor 分别使用独立端口）
- **评测集**：AIME 2024

#### 快速启动

具体的训练参数与启动方式均在 [`agentic/agentflow/`](./agentic/agentflow) 目录下。

---

## 框架接入方式

每个方法通过 slime 的三个自定义钩子接入训练流程：

```bash
--custom-generate-function-path  rollout.generate     # 自定义多步 rollout
--custom-rm-path                 rollout.reward_func  # 自定义 reward 计算
--custom-eval-rollout-log-function-path rollout.eval_log  # 自定义评测日志
```

`generate` 函数负责构造完整的 agent 轨迹并返回带 `loss_mask` 的 token 序列；仅 Planner 的生成 token 参与梯度计算，工具调用结果与最终输出的注入部分 mask 置 0。

---

## 环境依赖

推荐通过官方 Docker 镜像部署环境：

```bash
# 拉取最新镜像
docker pull slimerl/slime:latest

# 启动容器
docker run --rm --gpus all --ipc=host --shm-size=16g \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -it slimerl/slime:latest /bin/bash
```

手动安装请参见项目根目录 [`requirements.txt`](./requirements.txt) 及 [`docker/`](./docker)。

核心依赖：

- `slime >= 0.2.2`
- `sglang`
- `ray`
- `transformers`
- `torch >= 2.0`

---

## 计划复现


欢迎 PR 或 Issue 讨论新方法的接入。

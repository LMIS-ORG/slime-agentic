from pydantic import BaseModel
import re
from typing import Tuple

# Planner: QueryAnalysis
class QueryAnalysis(BaseModel):
    concise_summary: str
    required_skills: str
    relevant_tools: str
    additional_considerations: str

    def __str__(self):
        return f"""
Concise Summary: {self.concise_summary}

Required Skills:
{self.required_skills}

Relevant Tools:
{self.relevant_tools}

Additional Considerations:
{self.additional_considerations}
"""

# Planner: NextStep
class NextStep(BaseModel):
    justification: str
    context: str
    sub_goal: str
    tool_name: str


def extract_context_subgoal_and_tool(raw_response: str) -> Tuple[str, str, str]:
    """
    从 LLM 原始 response 中抽取 Context / Sub-Goal / Tool Name。
    参考已有实现，兼容 Markdown 粗体和多行内容，格式大致为：
      **Justification:** ...
      **Context:** ...
      **Sub-Goal:** ...
      **Tool Name:** ...
    """

    text = raw_response

    # 只保留最后一段 assistant（防止前面还有系统 / 用户模板残留）
    if "<|im_start|>assistant" in text:
        text = text.split("<|im_start|>assistant")[-1]

    # 去掉 <think> ... </think>，只保留正式回答部分
    if "</think>" in text:
        text = text.split("</think>", 1)[1]

    # 去掉 Markdown 粗体标记，方便用简洁模式匹配
    text = text.replace("**", "")

    # 匹配：
    #   Context:   <任意内容>
    #   Sub-Goal: <任意内容>
    #   Tool Name:<任意内容>
    pattern = r"Context:\s*(.*?)Sub-Goal:\s*(.*?)Tool Name:\s*(.*?)\s*(?:```)?\s*(?=\n\n|\Z)"
    matches = re.findall(pattern, text, re.DOTALL)
    if not matches:
        return "", "", ""

    # 使用最后一段匹配（通常是本次 step 的结果）
    context, sub_goal, tool_name = matches[-1]
    context = context.strip()
    sub_goal = sub_goal.strip()
    # 去掉前后空白，并裁掉尾部的特殊 token（例如 <|im_end|>）
    tool_name = tool_name.strip()
    if "<|im_end|>" in tool_name:
        tool_name = tool_name.split("<|im_end|>", 1)[0].strip()
    return context, sub_goal, tool_name

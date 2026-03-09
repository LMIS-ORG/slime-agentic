# Tool name mapping - this defines the external name for this tool
from __future__ import annotations

from tools.base import BaseTool
TOOL_NAME = "Generalist_Solution_Generator_Tool"

LIMITATION = f"""
The {TOOL_NAME} may provide hallucinated or incorrect responses.
"""

BEST_PRACTICE = f"""
For optimal results with the {TOOL_NAME}:
1. Use it for general queries or tasks that don't require specialized knowledge or specific tools in the toolbox.
2. Provide clear, specific query.   
3. Use it to answer the original query through step by step reasoning for tasks without complex or multi-step reasoning.
4. For complex queries, break them down into subtasks and use the tool multiple times.
5. Use it as a starting point for complex tasks, then refine with specialized tools.
6. Verify important information from its responses.
"""

TOOL_DESCRIPTION = f"""
A generalized tool that takes query from the user, and answers the question step by step to the best of its ability.
"""

TOOL_DEMO_COMMANDS = {                
                    "command": 'execution = tool.execute(query="Summarize the following text in a few lines")',
                    "description": "Generate a short summary given the query from the user."
                }


class Base_Generator_Tool(BaseTool):
    def __init__(self,llm_engine: SGLangEngine):
        super().__init__(
            tool_name=TOOL_NAME,
            tool_description=TOOL_DESCRIPTION,
            demo_commands=TOOL_DEMO_COMMANDS

        )
        self.llm_engine = llm_engine

    async def execute(self, query: str) -> str:
        messages = [{"role": "user", "content": query}]
        out = await self.llm_engine.generate(messages)
        return out.response


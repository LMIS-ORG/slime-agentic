import logging
import re
from typing import Any

from llm_engine import SGLangEngine
from memory import Memory

logger = logging.getLogger(__name__)


class Verifier:
    def __init__(self, llm_engine: SGLangEngine, available_tools: list[str], tools_metadata: dict):
        self.llm_engine = llm_engine
        self.available_tools = available_tools
        self.tools_metadata = tools_metadata

    async def verificate_context(self, question: str, query_analysis: str, step_count: int, memory: Memory) -> Any:

        prompt_memory_verification = f"""
Task: Evaluate if the current memory is complete and accurate enough to answer the query, or if more tools are needed.

Context:
- **Query:** {question}
- **Available Tools:** {self.available_tools}
- **Toolbox Metadata:** {self.tools_metadata}
- **Initial Analysis:** {query_analysis}
- **Memory (Tools Used & Results):** {memory.get_actions()}

Instructions:
1.  Review the query, initial analysis, and memory.
2.  Assess the completeness of the memory: Does it fully address all parts of the query?
3.  Check for potential issues:
    -   Are there any inconsistencies or contradictions?
    -   Is any information ambiguous or in need of verification?
4.  Determine if any unused tools could provide missing information.

Final Determination:
-   If the memory is sufficient, explain why and conclude with "STOP".
-   If more information is needed, explain what's missing, which tools could help, and conclude with "CONTINUE".

IMPORTANT: The response must end with either "Conclusion: STOP" or "Conclusion: CONTINUE".
"""
        messages = [{"role": "user", "content": prompt_memory_verification}]
        verifier_out = await self.llm_engine.generate(messages)
        logger.debug("verifier response: %s", verifier_out.response)
        analysis, conclusion = self.parse_conclusion(verifier_out.response)
        logger.debug("verifier conclusion: %s", conclusion)
        return analysis, conclusion, verifier_out

    def parse_conclusion(self, response: str) -> tuple[str, str]:
        analysis = response
        pattern = r'conclusion\**:?\s*\**\s*(STOP|CONTINUE)\b'
        matches = list(re.finditer(pattern, response, re.IGNORECASE))
        if matches:
            return analysis, matches[-1].group(1).upper()

        last_lines = response.strip().splitlines()[-3:]
        tail = "\n".join(last_lines).lower()
        if re.search(r'\bstop\b', tail):
            return analysis, 'STOP'
        elif re.search(r'\bcontinue\b', tail):
            return analysis, 'CONTINUE'
        else:
            logger.warning("No valid conclusion (STOP or CONTINUE) found in the response. Defaulting to CONTINUE.")
            return analysis, 'CONTINUE'

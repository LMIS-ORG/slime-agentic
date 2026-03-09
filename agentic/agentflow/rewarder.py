import re


class Rewarder:
    def __init__(self, llm_engine):
        self.llm_engine = llm_engine

    async def compute_reward(self, question: str, model_response: str, groundtruth: str) -> float:
        query_prompt = f"""You are a strict math answer evaluator.

**Task:** Read the Model Response, extract its final answer, and determine if it matches the Ground Truth.

**Steps:**
1. Read the Model Response carefully. Find the final answer — look for \\boxed{{...}}, "the answer is ...", "result: ...", or the last numerical conclusion.
2. Compare the extracted answer to the Ground Truth.
3. They are equivalent ONLY if they represent the same mathematical value (e.g., "1/2" == "0.5", "1,000" == "1000").
4. If the Model Response has no clear final answer, or the answer does not match, output False.
5. Do NOT be lenient. When in doubt, output False.

**Inputs:**
Question: {question}

Model Response:
{model_response}

Ground Truth: {groundtruth}

**You MUST end your response with exactly one of these two lines (no extra text after it):**
VERDICT: True
VERDICT: False"""

        messages = [{"role": "user", "content": query_prompt}]
        out = await self.llm_engine.generate(messages)
        response = out.response.strip()

        match = re.search(r"VERDICT\s*:\s*(True|False)", response, re.IGNORECASE)
        if match:
            return 1.0 if match.group(1).lower() == "true" else 0.0

        match = re.search(r"<true_false>\s*:?\s*(true|false)", response, re.IGNORECASE)
        if match:
            return 1.0 if match.group(1).lower() == "true" else 0.0

        last_line = response.strip().splitlines()[-1].strip().lower() if response.strip() else ""
        if last_line in ("true", "true.", "verdict: true"):
            return 1.0
        if last_line in ("false", "false.", "verdict: false"):
            return 0.0

        return 0.0

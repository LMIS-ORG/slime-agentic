from __future__ import annotations

import asyncio
import os
import re
import sys

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from tools.base import BaseTool

_EXEC_TIMEOUT = 30  # seconds

_DANGEROUS_CALLS = ["exit", "quit", "sys.exit", "os._exit"]

_MAX_OUTPUT_LENGTH = 4000

# Tool name mapping - this defines the external name for this tool
TOOL_NAME = "Python_Code_Generator_Tool"

LIMITATION = f"""
The {TOOL_NAME} has the following limitations:
1. Code execution is time-limited to {_EXEC_TIMEOUT} seconds; avoid infinite loops or extremely slow algorithms.
2. No GUI output (matplotlib.pyplot.show() has no effect); use print() to output results.
3. File and network I/O may not work in the sandbox environment.
"""

BEST_PRACTICE = f"""
For optimal results with the {TOOL_NAME}:
1. Describe the calculation or problem in plain language; do not include Python code in the query.
2. The tool will generate and run Python code automatically — just state what needs to be computed.
3. Always end the generated code with a print() statement so the result is captured.
"""

TOOL_DESCRIPTION = """
A tool that generates and executes Python code snippets for calculations and math-related problems.
It returns the printed output from the executed code.
"""

TOOL_DEMO_COMMANDS = [
    {
        "command": 'execution = tool.execute(query="Calculate the factorial of 5")',
        "description": "Generate a Python code snippet to calculate the factorial of 5.",
    },
    {
        "command": 'execution = tool.execute(query="Find the sum of prime numbers up to 50")',
        "description": "Generate a Python code snippet to find the sum of prime numbers up to 50.",
    },
    {
        "command": (
            'query="Given the list [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], '
            'calculate the sum of squares of odd numbers"\n'
            "execution = tool.execute(query=query)"
        ),
        "description": "Generate a Python function for a specific mathematical operation on a given list of numbers.",
    },
]


def _truncate(text: str, max_length: int) -> str:
    if len(text) <= max_length:
        return text
    half = max_length // 2 - 30
    return text[:half] + "\n... (truncated) ...\n" + text[-half:]


def _sanitize_code(code: str) -> str:
    """Remove dangerous calls (exit / quit / sys.exit) that would kill the process."""
    sanitized = code
    for func in _DANGEROUS_CALLS:
        sanitized = re.sub(rf"{re.escape(func)}\s*\([^)]*\)", "pass", sanitized)
    return sanitized


class Python_Coder_Tool(BaseTool):
    def __init__(self, llm_engine):
        super().__init__(
            tool_name=TOOL_NAME,
            tool_description=TOOL_DESCRIPTION,
            demo_commands=TOOL_DEMO_COMMANDS,
        )
        self.llm_engine = llm_engine

    @staticmethod
    def preprocess_code(code: str) -> str:
        """Extract the first Python code block from the LLM response.

        Falls back progressively:
        1. ```python ... ``` block
        2. Any ``` ... ``` block
        3. The raw response itself (treat entire reply as code)
        """
        match = re.search(r"```python\s*(.*?)\s*```", code, re.DOTALL)
        if match:
            return match.group(1).strip()
        match = re.search(r"```\s*(.*?)\s*```", code, re.DOTALL)
        if match:
            return match.group(1).strip()
        return code.strip()

    @staticmethod
    async def _run_code_in_subprocess(code: str, timeout: int) -> str:
        """Run code in a subprocess that can be killed cleanly on timeout."""
        proc = await asyncio.create_subprocess_exec(
            sys.executable, "-c", code,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=timeout,
            )
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            return f"Execution error: code timed out after {timeout}s"

        output = stdout.decode(errors="replace").strip()
        err_output = stderr.decode(errors="replace").strip()

        if proc.returncode != 0:
            return f"Execution error: {err_output or f'exit code {proc.returncode}'}"

        if not output:
            if err_output:
                return f"(no stdout, stderr: {_truncate(err_output, _MAX_OUTPUT_LENGTH)})"
            return "(no output)"

        return _truncate(output, _MAX_OUTPUT_LENGTH)

    async def execute(self, query: str) -> str:
        """Generate Python code for *query*, execute it, and return results.

        Never raises — all errors are returned as descriptive strings so the
        solver can continue without crashing.
        """
        # Step 1: Ask the LLM to generate Python code
        try:
            system_prompt = (
                "You are a Python code generator. "
                "Write a self-contained Python script that solves the problem and prints the final result.\n"
                f"{LIMITATION}\n"
                f"{BEST_PRACTICE}\n"
                "Return ONLY a single Python code block wrapped in ```python ... ```."
            )
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query},
            ]
            out = await self.llm_engine.generate(messages)
        except Exception as exc:
            return f"Code generation error: {exc}"

        # Step 2: Extract & sanitize code
        code = self.preprocess_code(out.response)
        code = _sanitize_code(code)

        # Step 3: Execute in a subprocess (can be killed cleanly on timeout)
        try:
            result = await self._run_code_in_subprocess(code, _EXEC_TIMEOUT)
        except Exception as exc:
            result = f"Execution error: {exc}"

        return result

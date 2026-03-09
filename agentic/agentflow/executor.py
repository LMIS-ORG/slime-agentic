import importlib.util
import re
import sys
import warnings
from pathlib import Path
from typing import Any

from llm_engine import SGLangEngine

# Tool name mapping: Static fallback mapping (long external names to internal)
TOOL_NAME_MAPPING_LONG = {
    "Generalist_Solution_Generator_Tool": {
        "class_name": "Base_Generator_Tool",
        "dir_name": "base_generator"
    },
    "Python_Code_Generator_Tool": {
        "class_name": "Python_Coder_Tool",
        "dir_name": "python_coder"
    },
}

# Short to long mapping for fallback
TOOL_NAME_MAPPING_SHORT = {
    "Base_Generator_Tool": "Generalist_Solution_Generator_Tool",
    "Python_Coder_Tool": "Python_Code_Generator_Tool",
}

class Executor:
    def __init__(
        self,
        llm_engine: SGLangEngine,
        available_tools: list[str],
        tools_engine_map: dict[str, Any] | None = None,
    ):
        self.llm_engine = llm_engine
        # key 可以是 TOOL_NAME（外部名）、class_name 或 dir_name，找不到则 fallback 到 llm_engine
        self.tools_engine_map: dict[str, Any] = tools_engine_map or {}
        self.available_tools = available_tools

    def extract_explanation_and_command(self, response: Any) -> tuple[str, str, str]:
        def normalize_code(code: str) -> str:
            normalized = code.strip()
            normalized = re.sub(r"<\|im_end\|>\s*$", "", normalized).strip()
            normalized = re.sub(r"^```python\s*", "", normalized, flags=re.IGNORECASE)
            normalized = re.sub(r"^```", "", normalized).strip()
            normalized = re.sub(r"```$", "", normalized).strip()
            normalized = re.sub(r"^\([^)]*pid=\d+[^)]*\)\s*", "", normalized)
            return normalized.strip()

        analysis = "No analysis found."
        explanation = "No explanation found."
        command = "No command found."

        if hasattr(response, "response"):
            response = response.response

        if isinstance(response, str):
            text = response.strip()

            analysis_match = re.search(
                r"Analysis:(.*?)Command Explanation",
                text,
                re.DOTALL | re.IGNORECASE,
            )
            if analysis_match:
                analysis = analysis_match.group(1).strip()

            explanation_match = re.search(
                r"Command Explanation:(.*?)Generated Command",
                text,
                re.DOTALL | re.IGNORECASE,
            )
            if explanation_match:
                explanation = explanation_match.group(1).strip()

            command_match = re.search(
                r"Generated Command:.*?```python\s*\n(.*?)```",
                text,
                re.DOTALL | re.IGNORECASE,
            )
            if command_match:
                command = command_match.group(1).strip()
            else:
                loose_matches = re.findall(
                    r"```python\s*\n(.*?)```",
                    text,
                    re.DOTALL | re.IGNORECASE,
                )
                if loose_matches:
                    command = max(loose_matches, key=lambda x: len(x.strip())).strip()
                else:
                    direct_match = re.search(
                        r"(execution\s*=\s*tool\.execute\(.*\))",
                        text,
                        re.DOTALL,
                    )
                    if direct_match:
                        command = direct_match.group(1).strip()
        else:
            command = "Invalid response type."

        command = normalize_code(command)
        #print("parsed_analysis: ", analysis)
        #print("parsed_explanation: ", explanation)
        #print("parsed_command: ", command)
        return analysis, explanation, command

    async def generate_tool_command(self,query: str,context: str,sub_goal: str,tool_name: str,tool_metadata: dict,step_count: int) -> str:
        prompt_generate_tool_command = f"""
        Task: Generate a precise command to execute the selected tool.

Context:
- **Query:** {query}
- **Sub-Goal:** {sub_goal}
- **Tool Name:** {tool_name}
- **Tool Metadata:** {tool_metadata}
- **Relevant Data:** {context}

Instructions:
1.  Analyze the tool's required parameters from its metadata.
2.  Construct valid Python code that addresses the sub-goal using the provided context and data.
3.  The command must include at least one call to `tool.execute()`.
4.  Each `tool.execute()` call must be assigned to a variable named **`execution`**.
5.  Please give the exact numbers and parameters should be used in the `tool.execute()` call.
6.  **IMPORTANT: `tool.execute()` only accepts a single keyword argument: `query`. Do NOT pass any other keyword arguments (e.g. no `context`, `data`, `input`, etc.).**
7.  **IMPORTANT: The `query` value MUST be a plain-language description of what to compute. Do NOT put Python code inside the query string.**
8.  **IMPORTANT: Always wrap the `query` value in triple double-quotes (`\"\"\"...\"\"\"`). This prevents syntax errors from special characters or apostrophes.**

Output Format:
Present your response in the following structured format. Do not include any extra text or explanations.

Generated Command:
```python
<command>
```

Example1:
Generated Command:
```python
execution = tool.execute(query=\"\"\"Calculate the factorial of 10\"\"\")
```

Example2:
Generated Command:
```python
execution = tool.execute(query=\"\"\"Find the number of intersections of y = 4*g(f(sin(2*pi*x))) and x = 4*g(f(cos(3*pi*y))), where f(x)=|x|-0.5 and g(x)=|x|-0.25\"\"\")
```
"""
        messages = [{"role": "user", "content": prompt_generate_tool_command}]
        tool_cmd_out = await self.llm_engine.generate(messages)
        analysis, explanation, command = self.extract_explanation_and_command(tool_cmd_out.response)
        return command, tool_cmd_out

    def _parse_command_kwargs(self, command: str) -> dict:
        """Extract kwargs by executing the command against a fake tool object.

        Falls back to regex extraction when the command is not valid Python
        (e.g. the LLM embedded raw Python code with mismatched quotes inside
        the query string).
        """
        captured: dict = {}

        class _FakeTool:
            def execute(self_, **kwargs):  # noqa: N805
                captured.update(kwargs)

        # Level 1: exec the command directly (works for well-formed commands)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", SyntaxWarning)
                exec(command, {"tool": _FakeTool()})  # noqa: S102
            if captured:
                return captured
        except Exception:
            pass

        # Level 2: regex extraction — try triple-quote delimiters first, then single/double
        for delim in ['"""', "'''", '"', "'"]:
            esc = re.escape(delim)
            m = re.search(
                rf'tool\.execute\(\s*query\s*=\s*{esc}(.*?){esc}\s*\)\s*$',
                command,
                re.DOTALL,
            )
            if m:
                return {"query": m.group(1)}

        # Level 3: greedy grab — take everything between query= and the last closing paren
        m = re.search(r'tool\.execute\(\s*query\s*=\s*([\s\S]+)\)\s*$', command)
        if m:
            val = m.group(1).strip()
            for delim in ['"""', "'''", '"', "'"]:
                if val.startswith(delim):
                    val = val[len(delim):]
                    if val.endswith(delim):
                        val = val[: -len(delim)]
                    break
            return {"query": val}

        raise ValueError(f"Cannot parse command: {command!r}")

    def _resolve_tool_mapping(self, tool_name: str) -> dict:
        """Resolve a tool name (TOOL_NAME / class_name / dir_name) to its mapping dict."""
        cleaned = tool_name.strip().strip("`").strip()

        # 1) Try TOOL_NAME (external long name)
        if cleaned in TOOL_NAME_MAPPING_LONG:
            return TOOL_NAME_MAPPING_LONG[cleaned]

        # 2) Try class_name → long name → mapping
        long_name = TOOL_NAME_MAPPING_SHORT.get(cleaned)
        if long_name and long_name in TOOL_NAME_MAPPING_LONG:
            return TOOL_NAME_MAPPING_LONG[long_name]

        # 3) Try dir_name (e.g. "python_coder", "base_generator")
        for mapping in TOOL_NAME_MAPPING_LONG.values():
            if mapping["dir_name"] == cleaned or mapping["class_name"] == cleaned:
                return mapping

        import logging
        logging.getLogger(__name__).warning(
            "Unknown tool_name: %r, falling back to Base_Generator_Tool.", tool_name,
        )
        return TOOL_NAME_MAPPING_LONG["Generalist_Solution_Generator_Tool"]

    def _load_tool(self, tool_name: str, tools_dir: str):
        """Dynamically load and instantiate a tool class given its external tool_name."""
        mapping = self._resolve_tool_mapping(tool_name)

        class_name = mapping["class_name"]
        dir_name = mapping["dir_name"]
        tool_file = Path(tools_dir) / dir_name / "tool.py"
        if not tool_file.exists():
            raise FileNotFoundError(f"Tool file not found: {tool_file}")

        module_key = f"_tool_{dir_name}"
        if module_key not in sys.modules:
            spec = importlib.util.spec_from_file_location(module_key, tool_file)
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_key] = module
            spec.loader.exec_module(module)
        else:
            module = sys.modules[module_key]

        cls = getattr(module, class_name)
        engine = self.tools_engine_map.get(dir_name, self.llm_engine)
        return cls(llm_engine=engine)

    async def execute_command(self, tool_name: str, command: str, tools_dir: str) -> Any:
        """Load the tool, parse the command kwargs, and call tool.execute(**kwargs).

        Never raises — returns an error string on failure so the solver can
        continue to the next step instead of crashing the entire rollout.
        """
        try:
            tool = self._load_tool(tool_name, tools_dir)
        except Exception as exc:
            return f"Tool load error ({tool_name}): {exc}"

        try:
            kwargs = self._parse_command_kwargs(command)
        except Exception as exc:
            return f"Command parse error: {exc}"

        try:
            execution = await tool.execute(**kwargs)
            return execution
        except Exception as exc:
            return f"Tool execution error ({tool_name}): {exc}"



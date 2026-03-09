class BaseTool:
    def __init__(self, tool_name=None, tool_description=None,demo_commands=None):
        self.tool_name = tool_name
        self.tool_description = tool_description
        self.demo_commands = demo_commands


from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable
from enum import Enum


class ToolError(Exception):
    pass


@dataclass
class ToolParameter:
    name: str
    type: str
    description: str
    required: bool = True
    enum: list[str] | None = None


@dataclass
class Tool:
    name: str
    description: str
    parameters: list[ToolParameter] = field(default_factory=list)
    handler: Callable[[dict], Any] | None = None
    schema_context: str | None = None

    def to_openai_format(self) -> dict:
        props = {}
        required = []

        for param in self.parameters:
            prop = {"type": param.type, "description": param.description}
            if param.enum:
                prop["enum"] = param.enum
            props[param.name] = prop
            if param.required:
                required.append(param.name)

        result = {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": props,
                },
            },
        }

        if required:
            result["function"]["parameters"]["required"] = required

        return result

    def validate_arguments(self, arguments: dict) -> tuple[bool, str]:
        for param in self.parameters:
            if param.required and param.name not in arguments:
                return False, f"Missing required parameter: {param.name}"

            if param.name in arguments:
                value = arguments[param.name]
                expected_type = param.type

                if expected_type == "string" and not isinstance(value, str):
                    return False, f"Parameter '{param.name}' must be a string"
                elif expected_type == "integer" and not isinstance(value, int):
                    return False, f"Parameter '{param.name}' must be an integer"
                elif expected_type == "number" and not isinstance(value, (int, float)):
                    return False, f"Parameter '{param.name}' must be a number"
                elif expected_type == "boolean" and not isinstance(value, bool):
                    return False, f"Parameter '{param.name}' must be a boolean"
                elif expected_type == "array" and not isinstance(value, list):
                    return False, f"Parameter '{param.name}' must be an array"
                elif expected_type == "object" and not isinstance(value, dict):
                    return False, f"Parameter '{param.name}' must be an object"
                elif expected_type == "any":
                    pass
                elif param.enum and value not in param.enum:
                    return False, f"Parameter '{param.name}' must be one of: {param.enum}"

        return True, ""

    def execute(self, arguments: dict) -> Any:
        valid, error_msg = self.validate_arguments(arguments)
        if not valid:
            raise ToolError(error_msg)

        if self.handler is None:
            raise ToolError(f"Tool '{self.name}' has no handler registered")

        return self.handler(arguments)


@dataclass
class ToolCall:
    tool_name: str
    arguments: dict
    raw_text: str = ""


class ToolCallStatus(Enum):
    NO_TOOL_CALL = "no_tool_call"
    INCOMPLETE = "incomplete"
    COMPLETE = "complete"
    ERROR = "error"


@dataclass
class ToolCallResult:
    call: ToolCall | None
    status: ToolCallStatus
    error_message: str | None = None
    output: Any = None


class ToolRegistry:
    TOOL_CALL_OPEN = "<tool_call>"
    TOOL_CALL_CLOSE = "</tool_call>"
    TOOL_RESPONSE_OPEN = "<tool_response>"
    TOOL_RESPONSE_CLOSE = "</tool_response>"

    def __init__(self):
        self._tools: dict[str, Tool] = {}
        self._on_tool_call: Callable[[str, dict], None] | None = None
        self._on_tool_result: Callable[[str, Any, str | None], None] | None = None

    def set_callbacks(
        self,
        on_tool_call: Callable[[str, dict], None] | None = None,
        on_tool_result: Callable[[str, Any, str | None], None] | None = None,
    ) -> None:
        """Set callbacks for tool events.

        Args:
            on_tool_call: Called when a tool is called with (tool_name, arguments)
            on_tool_result: Called when a tool result is ready with (tool_name, result, error)
        """
        self._on_tool_call = on_tool_call
        self._on_tool_result = on_tool_result

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"Tool '{tool.name}' is already registered")
        self._tools[tool.name] = tool

    def unregister(self, name: str) -> bool:
        if name in self._tools:
            del self._tools[name]
            return True
        return False

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def get_all_tools(self) -> list[Tool]:
        return list(self._tools.values())

    def get_tools_for_prompt(self) -> list[dict]:
        return [tool.to_openai_format() for tool in self._tools.values()]

    def format_tools_for_prompt(self) -> str:
        lines = ["## Available Tools", ""]

        for tool in self._tools.values():
            lines.append(f"### {tool.name}")
            lines.append(f"{tool.description}")
            lines.append("")
            lines.append("Parameters:")
            for param in tool.parameters:
                req_str = "(required)" if param.required else "(optional)"
                lines.append(f"  - {param.name}: {param.type} {req_str} - {param.description}")
            lines.append("")

        return "\n".join(lines)

    def parse_tool_calls(self, text: str) -> ToolCallResult:
        open_pattern = re.escape(self.TOOL_CALL_OPEN)
        close_pattern = re.escape(self.TOOL_CALL_CLOSE)

        pattern = rf"{open_pattern}\s*(.*?)\s*{close_pattern}"
        match = re.search(pattern, text, re.DOTALL)

        if not match:
            return ToolCallResult(call=None, status=ToolCallStatus.NO_TOOL_CALL)

        raw_text = match.group(1).strip()
        call = self._parse_tool_call_text(raw_text)

        if call is None:
            return ToolCallResult(
                call=None,
                status=ToolCallStatus.INCOMPLETE,
                error_message="Incomplete tool call"
            )

        if call.tool_name not in self._tools:
            return ToolCallResult(
                call=call,
                status=ToolCallStatus.ERROR,
                error_message=f"Unknown tool: {call.tool_name}"
            )

        return ToolCallResult(call=call, status=ToolCallStatus.COMPLETE)

    def _parse_tool_call_text(self, raw_text: str) -> ToolCall | None:
        lines = raw_text.strip().split("\n")
        if not lines:
            return None

        tool_name = None
        arguments_text = ""

        in_arguments = False
        for line in lines:
            line = line.strip()

            if line.startswith("name:"):
                tool_name = line[5:].strip().strip('"').strip("'")
            elif line.startswith("arguments:") or line.startswith("args:"):
                in_arguments = True
                arguments_text = line.split(":", 1)[1].strip()
            elif in_arguments:
                arguments_text += "\n" + line

        if not tool_name:
            return None

        try:
            if arguments_text.strip():
                if arguments_text.strip().startswith("{"):
                    arguments = json.loads(arguments_text)
                else:
                    arguments = json.loads("{" + arguments_text + "}")
            else:
                arguments = {}
        except json.JSONDecodeError:
            arguments = {}

        return ToolCall(tool_name=tool_name, arguments=arguments, raw_text=raw_text)

    def execute_tool_call(self, call: ToolCall) -> ToolCallResult:
        tool = self._tools.get(call.tool_name)
        if not tool:
            return ToolCallResult(
                call=call,
                status=ToolCallStatus.ERROR,
                error_message=f"Unknown tool: {call.tool_name}"
            )

        if self._on_tool_call:
            self._on_tool_call(call.tool_name, call.arguments)

        try:
            result = tool.execute(call.arguments)
            if self._on_tool_result:
                self._on_tool_result(call.tool_name, result, None)
            return ToolCallResult(
                call=call,
                status=ToolCallStatus.COMPLETE,
                output=result
            )
        except ToolError as e:
            if self._on_tool_result:
                self._on_tool_result(call.tool_name, None, str(e))
            return ToolCallResult(
                call=call,
                status=ToolCallStatus.ERROR,
                error_message=str(e)
            )
        except Exception as e:
            error_msg = f"Execution error: {str(e)}"
            if self._on_tool_result:
                self._on_tool_result(call.tool_name, None, error_msg)
            return ToolCallResult(
                call=call,
                status=ToolCallStatus.ERROR,
                error_message=error_msg
            )

    def format_tool_response(self, result: ToolCallResult) -> str:
        if result.status == ToolCallStatus.ERROR:
            response_content = json.dumps({"error": result.error_message})
        else:
            if isinstance(result.output, str):
                response_content = json.dumps(result.output)
            else:
                response_content = json.dumps(result.output, indent=2)

        return (
            f"{self.TOOL_RESPONSE_OPEN}\n"
            f"{response_content}\n"
            f"{self.TOOL_RESPONSE_CLOSE}"
        )


def create_string_param(
    name: str,
    description: str,
    required: bool = True,
    enum: list[str] | None = None,
) -> ToolParameter:
    return ToolParameter(name=name, type="string", description=description, required=required, enum=enum)


def create_integer_param(name: str, description: str, required: bool = True) -> ToolParameter:
    return ToolParameter(name=name, type="integer", description=description, required=required)


def create_boolean_param(name: str, description: str, required: bool = True) -> ToolParameter:
    return ToolParameter(name=name, type="boolean", description=description, required=required)


def create_array_param(
    name: str,
    description: str,
    required: bool = True,
) -> ToolParameter:
    return ToolParameter(name=name, type="array", description=description, required=required)


def create_object_param(
    name: str,
    description: str,
    required: bool = True,
) -> ToolParameter:
    return ToolParameter(name=name, type="object", description=description, required=required)

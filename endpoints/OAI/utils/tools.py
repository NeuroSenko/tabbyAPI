import json
import re
import xml.etree.ElementTree as ET
from loguru import logger
from typing import List

from endpoints.OAI.types.tools import ToolCall


TOOL_CALL_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "function": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "arguments": {
                        # Converted to OAI's string in post process
                        "type": "object"
                    },
                },
                "required": ["name", "arguments"],
            },
        },
        "required": ["function"],
    },
}

# Qwen-style schema: {"name": "...", "arguments": {...}}
TOOL_CALL_SCHEMA_QWEN = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "arguments": {"type": "object"},
    },
    "required": ["name", "arguments"],
}


class ToolCallProcessor:
    @staticmethod
    def from_json(tool_calls_str: str) -> List[ToolCall]:
        """Postprocess tool call JSON to a parseable class

        Supports two formats:
        1. Array format: [{"function": {"name": "...", "arguments": {...}}}]
        2. Qwen format: {"name": "...", "arguments": {...}}
        """

        parsed = json.loads(tool_calls_str)

        # Check if it's Qwen format (single object with "name" and "arguments")
        if isinstance(parsed, dict) and "name" in parsed and "arguments" in parsed:
            # Convert Qwen format to standard format
            tool_call_dict = {
                "function": {
                    "name": parsed["name"],
                    "arguments": json.dumps(parsed["arguments"])
                    if isinstance(parsed["arguments"], dict)
                    else parsed["arguments"],
                }
            }
            return [ToolCall(**tool_call_dict)]

        # Standard array format
        tool_calls = parsed if isinstance(parsed, list) else [parsed]
        for tool_call in tool_calls:
            tool_call["function"]["arguments"] = json.dumps(
                tool_call["function"]["arguments"]
            )

        return [ToolCall(**tool_call) for tool_call in tool_calls]

    @staticmethod
    def from_xml(tool_calls_str: str) -> List[ToolCall]:
        """Parse tool calls from XML format to ToolCall objects

        Supports three XML formats:
        1. <invoke name="tool"><parameter name="arg">value</parameter></invoke>
        2. <tool_call>{"name": "tool", "arguments": {...}}</tool_call>
        3. <tool_call><function=tool><parameter=arg>value</parameter></function></tool_call>
        """

        tool_calls = []

        # First, try the <tool_call> format with embedded JSON
        tool_call_pattern = r'<tool_call>\s*(\{.*?\})\s*</tool_call>'
        tool_call_matches = re.findall(tool_call_pattern, tool_calls_str, re.DOTALL)

        if tool_call_matches:
            # Parse <tool_call> format with JSON
            for json_str in tool_call_matches:
                try:
                    tool_data = json.loads(json_str)
                    tool_call_dict = {
                        "function": {
                            "name": tool_data.get("name"),
                            "arguments": json.dumps(tool_data.get("arguments", {}))
                            if not isinstance(tool_data.get("arguments"), str)
                            else tool_data.get("arguments"),
                        }
                    }
                    tool_calls.append(ToolCall(**tool_call_dict))
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse tool_call JSON: {e}")
        else:
            # Try the Qwen-style format: <tool_call><function=name><parameter=arg>value</parameter></function></tool_call>
            qwen_tool_call_pattern = r'<tool_call>\s*<function=([^>]+)>(.*?)</function>\s*</tool_call>'
            qwen_param_pattern = r'<parameter=([^>]+)>(.*?)</parameter>'

            qwen_matches = re.findall(qwen_tool_call_pattern, tool_calls_str, re.DOTALL)

            if qwen_matches:
                # Parse Qwen-style format
                for tool_name, function_body in qwen_matches:
                    parameters = {}
                    params = re.findall(qwen_param_pattern, function_body, re.DOTALL)

                    for param_name, param_value in params:
                        # Try to parse JSON values
                        param_value = param_value.strip()
                        try:
                            # Try parsing as JSON (for arrays, objects, booleans, numbers)
                            parsed_value = json.loads(param_value)
                            parameters[param_name] = parsed_value
                        except json.JSONDecodeError:
                            # Keep as string if not valid JSON
                            parameters[param_name] = param_value

                    # Convert to the expected format
                    tool_call_dict = {
                        "function": {
                            "name": tool_name,
                            "arguments": json.dumps(parameters),
                        }
                    }

                    tool_calls.append(ToolCall(**tool_call_dict))
            else:
                # Try the <invoke> format with <parameter> tags
                invoke_pattern = r'<invoke name="([^"]+)">(.*?)</invoke>'
                param_pattern = r'<parameter name="([^"]+)">(.*?)</parameter>'

                invocations = re.findall(invoke_pattern, tool_calls_str, re.DOTALL)

                for tool_name, invoke_body in invocations:
                    # Extract all parameters from the invoke body
                    parameters = {}
                    params = re.findall(param_pattern, invoke_body, re.DOTALL)

                    for param_name, param_value in params:
                        # Try to parse JSON values
                        param_value = param_value.strip()
                        try:
                            # Try parsing as JSON (for arrays, objects, booleans, numbers)
                            parsed_value = json.loads(param_value)
                            parameters[param_name] = parsed_value
                        except json.JSONDecodeError:
                            # Keep as string if not valid JSON
                            parameters[param_name] = param_value

                    # Convert to the expected format
                    tool_call_dict = {
                        "function": {
                            "name": tool_name,
                            "arguments": json.dumps(parameters),
                        }
                    }

                    tool_calls.append(ToolCall(**tool_call_dict))

        return tool_calls

    @staticmethod
    def dump(tool_calls: List[ToolCall]) -> List[dict]:
        """
        Convert ToolCall objects to a list of dictionaries.

        Args:
            tool_calls (List[ToolCall]): List of ToolCall objects to convert

        Returns:
            List[dict]: List of dictionaries representing the tool calls
        """

        # Don't use list comprehension here
        # as that will fail rather than warn
        dumped_tool_calls = []
        for tool_call_obj in tool_calls:
            try:
                dumped_tool_calls.append(tool_call_obj.model_dump())
            except (json.JSONDecodeError, AttributeError) as e:
                logger.warning(f"Error processing tool call: {e}")
        return dumped_tool_calls

    @staticmethod
    def to_json(tool_calls: List[ToolCall]) -> str:
        """
        Convert ToolCall objects to JSON string representation.

        Args:
            tool_calls (List[ToolCall]): List of ToolCall objects to convert

        Returns:
            str: JSON representation of the tool calls
        """

        if not tool_calls:
            return ""

        # Use the dump method to get the list of dictionaries
        dumped_tool_calls = ToolCallProcessor.dump(tool_calls)

        # Serialize the dumped array
        return json.dumps(dumped_tool_calls, indent=2)

    @staticmethod
    def to_xml(tool_calls: List[ToolCall], format: str = "claude") -> str:
        """
        Convert ToolCall objects to XML string representation.

        Args:
            tool_calls (List[ToolCall]): List of ToolCall objects to convert
            format (str): XML format to use - "claude" or "qwen" (default: "claude")

        Returns:
            str: XML representation of the tool calls
        """

        if not tool_calls:
            return ""

        xml_parts = []

        for tool_call_obj in tool_calls:
            try:
                # Parse the arguments JSON string
                arguments = json.loads(tool_call_obj.function.arguments)

                if format == "qwen":
                    # Build Qwen-style format: <tool_call><function=name><parameter=arg>value</parameter></function></tool_call>
                    qwen_lines = ['<tool_call>']
                    qwen_lines.append(f'<function={tool_call_obj.function.name}>')

                    # Add each parameter
                    for param_name, param_value in arguments.items():
                        # Convert value to string representation
                        if isinstance(param_value, str):
                            value_str = param_value
                        else:
                            # For complex types, use JSON representation
                            value_str = json.dumps(param_value)

                        qwen_lines.append(f'<parameter={param_name}>')
                        qwen_lines.append(value_str)
                        qwen_lines.append('</parameter>')

                    qwen_lines.append('</function>')
                    qwen_lines.append('</tool_call>')

                    xml_parts.append('\n'.join(qwen_lines))
                else:
                    # Build Claude-style format: <invoke name="tool"><parameter name="arg">value</parameter></invoke>
                    invoke_lines = [f'<invoke name="{tool_call_obj.function.name}">']

                    # Add each parameter
                    for param_name, param_value in arguments.items():
                        # Convert value to string representation
                        if isinstance(param_value, str):
                            value_str = param_value
                        else:
                            # For complex types, use JSON representation
                            value_str = json.dumps(param_value)

                        invoke_lines.append(f'<parameter name="{param_name}">{value_str}</parameter>')

                    invoke_lines.append('</invoke>')

                    xml_parts.append('\n'.join(invoke_lines))

            except (json.JSONDecodeError, AttributeError) as e:
                logger.warning(f"Error processing tool call for XML conversion: {e}")

        return '\n'.join(xml_parts)

"""Conversions between upstream DRIFT strings and current AgentDojo types."""

from __future__ import annotations

import ast
import json
import re
from collections.abc import Sequence
from typing import Any

from agentdojo.ast_utils import ASTParsingError, parse_tool_calls_from_python_function
from agentdojo.functions_runtime import FunctionCall, FunctionsRuntime
from agentdojo.types import ChatMessage, get_text_content_as_str


def tool_docs(runtime: FunctionsRuntime) -> list[dict[str, Any]]:
    return [
        {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters.model_json_schema(),
        }
        for tool in runtime.functions.values()
    ]


def message_text(message: ChatMessage) -> str:
    content = message["content"]
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    return get_text_content_as_str(content) or ""


def conversation_for_model(messages: Sequence[ChatMessage]) -> list[dict[str, Any]]:
    """Convert current AgentDojo messages to the history expected by DRIFT."""
    converted: list[dict[str, Any]] = []
    for message in messages:
        text = message_text(message)
        if message["role"] == "system":
            continue
        if message["role"] == "user":
            converted.append({"role": "user", "content": text})
        elif message["role"] == "assistant":
            calls = message.get("tool_calls") or []
            converted_message: dict[str, Any] = {
                "role": "assistant",
                "content": text or None,
            }
            if calls:
                converted_message["tool_calls"] = [
                    {
                        "id": call.id,
                        "type": "function",
                        "function": {
                            "name": call.function,
                            "arguments": json.dumps(call.args, ensure_ascii=False),
                        },
                    }
                    for call in calls
                ]
            converted.append(converted_message)
        elif message["role"] == "tool":
            converted.append(
                {
                    "role": "tool",
                    "content": text,
                    "tool_call_id": message["tool_call_id"] or "",
                }
            )
    return converted


def extract_tag(text: str, tag: str) -> str | None:
    match = re.search(rf"<{tag}>(.*?)</{tag}>", text, re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else None


def parse_trajectory(text: str) -> list[str]:
    # Upstream first accepts the nested <Traj-1> form, then falls back to
    # the outer <function_trajectory> representation.
    value = extract_tag(text, "Traj-1")
    if value is None:
        value = extract_tag(text, "function_trajectory")
    if value is None:
        return []
    return [name.strip().strip("'\"") for name in value.strip("[]").split(",") if name.strip()]


def parse_checklist(text: str) -> list[dict[str, Any]]:
    value = extract_tag(text, "parameter_checklist")
    if value is None:
        return []
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        try:
            parsed = ast.literal_eval(value)
        except (SyntaxError, ValueError):
            return []
    return parsed if isinstance(parsed, list) else []


def parse_agent_output(text: str) -> tuple[str, list[FunctionCall]]:
    thought = extract_tag(text, "function_thought") or ""
    final_answer = extract_tag(text, "final_answer") or ""
    raw_calls = extract_tag(text, "function_call") or "[]"
    call_source = raw_calls.strip()
    if not call_source:
        call_source = "[]"
    elif not call_source.startswith("["):
        call_source = f"[{call_source}]"
    try:
        calls = parse_tool_calls_from_python_function(call_source)
    except (ASTParsingError, IndexError, SyntaxError, ValueError):
        calls = []
    content = (
        f"<function_thought>{thought}</function_thought>\n\n"
        f"<function_call>{raw_calls}</function_call>\n\n"
        f"<final_answer>{final_answer}</final_answer>"
    )
    return content, calls


def remove_detected_instruction(text: str, instruction: str) -> str:
    words = instruction.split()
    if not words:
        return text
    pattern = r"\s*" + r"[\s\\]+".join(re.escape(word) for word in words) + r"\s*"
    return re.sub(pattern, " ", text, flags=re.DOTALL).strip()

"""Conversions between upstream DRIFT's string messages and current AgentDojo types."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

from agentdojo.functions_runtime import Function, FunctionCall, FunctionsRuntime
from agentdojo.types import ChatMessage, get_text_content_as_str, text_content_block_from_string


def message_text(message: ChatMessage) -> str:
    content = message["content"]
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    return get_text_content_as_str(content) or ""


def tool_docs(runtime: FunctionsRuntime) -> list[dict[str, Any]]:
    return [
        {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters.model_json_schema(),
        }
        for tool in runtime.functions.values()
    ]


def to_upstream_messages(messages: Sequence[ChatMessage]) -> list[dict[str, str]]:
    converted: list[dict[str, str]] = []
    for message in messages:
        role = message["role"]
        text = message_text(message)
        if role == "system":
            converted.append({"role": "system", "content": text})
        elif role == "user":
            converted.append({"role": "user", "content": text})
        elif role == "assistant":
            calls = message.get("tool_calls") or []
            if calls:
                rendered = ", ".join(
                    f"{call.function}({json.dumps(call.args, ensure_ascii=False)})" for call in calls
                )
                text = f"{text}\nPrevious function calls: {rendered}".strip()
            converted.append({"role": "assistant", "content": text})
        elif role == "tool":
            # Upstream DRIFT intentionally presents observations as user messages.
            converted.append({"role": "user", "content": text})
    return converted


def assistant_message(content: str, calls: list[FunctionCall]) -> ChatMessage:
    return {
        "role": "assistant",
        "content": [text_content_block_from_string(content)] if content else None,
        "tool_calls": calls or None,
    }


def call_error_message(content: str) -> ChatMessage:
    return {
        "role": "user",
        "content": [text_content_block_from_string(content)],
    }

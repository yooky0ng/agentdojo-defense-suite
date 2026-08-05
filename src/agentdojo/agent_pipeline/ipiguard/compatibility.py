"""AgentDojo 0.1.35 compatibility helpers for the upstream IPIGuard port."""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from openai.types.chat import ChatCompletionMessageParam

from agentdojo.agent_pipeline.llms.openai_llm import _message_to_openai
from agentdojo.functions_runtime import Function, FunctionCall
from agentdojo.types import (
    ChatAssistantMessage as AgentDojoChatAssistantMessage,
)
from agentdojo.types import (
    ChatMessage,
    MessageContentBlock,
    get_text_content_as_str,
    text_content_block_from_string,
)
from agentdojo.types import (
    ChatSystemMessage as AgentDojoChatSystemMessage,
)
from agentdojo.types import (
    ChatToolResultMessage as AgentDojoChatToolResultMessage,
)
from agentdojo.types import (
    ChatUserMessage as AgentDojoChatUserMessage,
)


def message_text(message: ChatMessage) -> str:
    content = message["content"]
    if content is None:
        return ""
    return get_text_content_as_str(content) or ""


def append_message_text(message: AgentDojoChatUserMessage, content: str) -> None:
    message["content"].append(text_content_block_from_string(content))


def ChatUserMessage(*, role: str, content: str) -> AgentDojoChatUserMessage:
    return AgentDojoChatUserMessage(role="user", content=[text_content_block_from_string(content)])


def ChatSystemMessage(*, role: str, content: str) -> AgentDojoChatSystemMessage:
    return AgentDojoChatSystemMessage(role="system", content=[text_content_block_from_string(content)])


def ChatAssistantMessage(
    *, role: str, content: str | None, tool_calls: list[FunctionCall] | None
) -> AgentDojoChatAssistantMessage:
    blocks: list[MessageContentBlock] | None = None if content is None else [text_content_block_from_string(content)]
    return AgentDojoChatAssistantMessage(role="assistant", content=blocks, tool_calls=tool_calls)


def ChatToolResultMessage(
    *, role: str, content: str, tool_call_id: str | None, tool_call: FunctionCall, error: str | None
) -> AgentDojoChatToolResultMessage:
    return AgentDojoChatToolResultMessage(
        role="tool",
        content=[text_content_block_from_string(content)],
        tool_call_id=tool_call.id,
        tool_call=tool_call,
        error=error,
    )


def openai_messages(messages: Sequence[ChatMessage], model: str) -> list[ChatCompletionMessageParam]:
    return [_message_to_openai(message, model) for message in messages]


def tool_docs(tools: Sequence[Function]) -> str:
    docs = ""
    for index, tool in enumerate(tools, start=1):
        definition: dict[str, Any] = {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters.model_json_schema(),
        }
        docs += f"<function-{index}>\n{json.dumps(definition, indent=4)}\n</function-{index}>\n\n"
    return docs


def tool_call_to_str(tool_call: FunctionCall, error: str | None = None) -> str:
    value: dict[str, Any] = {
        "function": tool_call.function,
        "args": dict(tool_call.args),
        "id": tool_call.id,
    }
    if error:
        value["error"] = error
    return json.dumps(value, indent=2)


def tool_returned_data_to_str(message: AgentDojoChatToolResultMessage) -> str:
    value = {
        "function": message["tool_call"].function,
        "returned_data": message_text(message),
        "id": message["tool_call_id"],
    }
    return json.dumps(value, indent=2)

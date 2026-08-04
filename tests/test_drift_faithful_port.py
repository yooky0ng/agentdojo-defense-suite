from __future__ import annotations

from types import SimpleNamespace

from agentdojo.agent_pipeline.drift import DRIFTLLM
from agentdojo.agent_pipeline.drift_simplified import (
    ENVIRONMENT_GUIDELINES,
    DRIFTClient,
    _conversation_for_model,
)
from agentdojo.functions_runtime import FunctionCall
from agentdojo.types import text_content_block_from_string


class _Completions:
    def __init__(self, content: str = "ok") -> None:
        self.content = content
        self.requests: list[dict] = []

    def create(self, **kwargs):
        self.requests.append(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=self.content))]
        )


class _OpenAI:
    def __init__(self) -> None:
        self.completions = _Completions()
        self.chat = SimpleNamespace(completions=self.completions)


class _PromptClient:
    def __init__(self, answers: list[str]) -> None:
        self.answers = iter(answers)
        self.calls = 0

    def run(self, system: str, user: str) -> str:
        self.calls += 1
        return next(self.answers)


def test_client_preserves_upstream_completion_limit() -> None:
    openai = _OpenAI()
    client = DRIFTClient(openai, "gpt-test")

    assert client.run("system", "user") == "ok"
    assert openai.completions.requests[0]["max_completion_tokens"] == 10000
    assert "temperature" not in openai.completions.requests[0]


def test_conversation_preserves_openai_tool_message_chain() -> None:
    call = FunctionCall(function="read_file", args={"file_path": "bill.txt"}, id="call-1")
    converted = _conversation_for_model(
        [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [call],
            },
            {
                "role": "tool",
                "content": [text_content_block_from_string("bill contents")],
                "tool_call_id": "call-1",
                "tool_call": call,
                "error": None,
            },
        ]
    )

    assert converted[0]["role"] == "assistant"
    assert converted[0]["tool_calls"][0]["function"]["name"] == "read_file"
    assert converted[1] == {
        "role": "tool",
        "content": "bill contents",
        "tool_call_id": "call-1",
    }


def test_checklist_formatting_retries_and_updates_state() -> None:
    client = _PromptClient(
        [
            "not valid json",
            '[{"name":"send_money","required parameters":null,"conditions":null}]',
        ]
    )
    drift = DRIFTLLM(client)  # type: ignore[arg-type]
    drift.node_checklist = [{"name": "send_money"}]

    drift._format_node_checklist("pay the bill")

    assert client.calls == 2
    assert drift.node_checklist == [
        {
            "name": "send_money",
            "required parameters": None,
            "conditions": None,
        }
    ]


def test_environment_guidelines_match_upstream_safety_contract() -> None:
    assert "try with a different query" in ENVIRONMENT_GUIDELINES
    assert "Do not assume the current year" in ENVIRONMENT_GUIDELINES
    assert "without requesting user confirmation" in ENVIRONMENT_GUIDELINES

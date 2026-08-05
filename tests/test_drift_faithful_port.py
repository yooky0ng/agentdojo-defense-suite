from __future__ import annotations

from types import SimpleNamespace

from agentdojo.agent_pipeline.drift import DRIFTLLM, DRIFTClient, DRIFTConfig
from agentdojo.agent_pipeline.drift.client import ENVIRONMENT_GUIDELINES
from agentdojo.agent_pipeline.drift.compatibility import conversation_for_model, parse_trajectory
from agentdojo.functions_runtime import FunctionCall, FunctionsRuntime
from agentdojo.types import ChatMessage, text_content_block_from_string


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
        self.requests: list[tuple[str, str]] = []

    def run(self, system: str, user: str) -> str:
        self.calls += 1
        self.requests.append((system, user))
        return next(self.answers)

    def run_conversation(self, system: str, messages: list[dict]) -> str:
        return '<function_call>[send_money(recipient="Mallory")]</function_call>'


def test_client_preserves_upstream_completion_limit() -> None:
    openai = _OpenAI()
    client = DRIFTClient(openai, "gpt-test")  # type: ignore[arg-type]

    assert client.run("system", "user") == "ok"
    assert openai.completions.requests[0]["max_completion_tokens"] == 10000
    assert "temperature" not in openai.completions.requests[0]


def test_conversation_preserves_openai_tool_message_chain() -> None:
    call = FunctionCall(function="read_file", args={"file_path": "bill.txt"}, id="call-1")
    converted = conversation_for_model(
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

    assert drift.node_json_formatting("pay the bill")

    assert client.calls == 2
    assert drift.node_checklist == [
        {
            "name": "send_money",
            "required parameters": None,
            "conditions": None,
        }
    ]


def test_trajectory_parser_accepts_upstream_traj_1_format() -> None:
    assert parse_trajectory(
        "<function_trajectory><Traj-1>[read_file, send_money]</Traj-1></function_trajectory>"
    ) == ["read_file", "send_money"]


def test_alignment_judge_uses_current_approved_trajectory() -> None:
    client = _PromptClient(["<Judge Result>Yes</Judge Result>"])
    drift = DRIFTLLM(client)  # type: ignore[arg-type]
    drift.function_trajectory = ["read_file", "send_money"]
    drift.achieved_function_trajectory = ["read_file"]
    drift.node_checklist = [
        {"name": "read_file", "required parameters": None, "conditions": None},
        {"name": "send_money", "required parameters": None, "conditions": None},
    ]
    drift.tool_permissions["get_contact"] = "Execute"

    assert drift.trajectory_constraint_validation("pay Alice", ["get_contact"])
    assert "['read_file', 'send_money']" in client.requests[0][1]


def test_trajectory_validation_fails_open_like_upstream() -> None:
    client = _PromptClient([])
    drift = DRIFTLLM(client)  # type: ignore[arg-type]
    drift.tool_permissions["send_money"] = "Execute"

    assert drift.trajectory_constraint_validation("pay Alice", ["send_money"])


def test_invalid_checklist_formatting_preserves_upstream_fail_open_signal() -> None:
    client = _PromptClient(["bad", "still bad", "also bad"])
    drift = DRIFTLLM(client)  # type: ignore[arg-type]
    drift.node_checklist = [{"name": "send_money"}]

    assert not drift.node_json_formatting("pay the bill")
    assert drift.node_checklist == [{"name": "send_money"}]


def test_query_stops_tool_calls_after_upstream_twenty_message_limit() -> None:
    client = _PromptClient([])
    drift = DRIFTLLM(
        client,  # type: ignore[arg-type]
        DRIFTConfig(
            build_constraints=False,
            injection_isolation=False,
            dynamic_validation=False,
        ),
    )
    messages: list[ChatMessage] = [
        {"role": "user", "content": [text_content_block_from_string(f"message {index}")]}
        for index in range(20)
    ]

    *_, returned, _ = drift.query("pay Alice", FunctionsRuntime(), messages=messages)

    assert returned[-1]["role"] == "assistant"
    assert returned[-1].get("tool_calls") is None


def test_environment_guidelines_match_upstream_safety_contract() -> None:
    assert "try with a different query" in ENVIRONMENT_GUIDELINES
    assert "Do not assume the current year" in ENVIRONMENT_GUIDELINES
    assert "without requesting user confirmation" in ENVIRONMENT_GUIDELINES

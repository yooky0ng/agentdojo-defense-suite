from collections.abc import Sequence

from agentdojo.agent_pipeline import AgentPipeline, PipelineConfig
from agentdojo.agent_pipeline.base_pipeline_element import BasePipelineElement
from agentdojo.defenses.task_shield.core import TaskShield, TaskShieldModel
from agentdojo.defenses.task_shield.pipeline import TaskShieldLLM, TaskShieldToolsExecutor
from agentdojo.functions_runtime import EmptyEnv, Env, FunctionCall, FunctionsRuntime, make_function
from agentdojo.types import ChatAssistantMessage, ChatMessage, text_content_block_from_string


class FakeShieldModel(TaskShieldModel):
    def __init__(self, responses: list[str]) -> None:
        self.responses = iter(responses)

    def generate(self, system_prompt: str, user_prompt: str, history: str = "") -> str:
        return next(self.responses)


def echo(value: str) -> str:
    """Return the provided value.

    :param value: The value to return.
    """
    return value


class FakeLLM(BasePipelineElement):
    def __init__(self, contents: list[str]) -> None:
        self.contents = iter(contents)
        self.name = "fake"

    def query(
        self,
        query: str,
        runtime: FunctionsRuntime,
        env: Env = EmptyEnv(),
        messages: Sequence[ChatMessage] = [],
        extra_args: dict = {},
    ) -> tuple[str, FunctionsRuntime, Env, Sequence[ChatMessage], dict]:
        message = ChatAssistantMessage(
            role="assistant",
            content=[text_content_block_from_string(next(self.contents))],
            tool_calls=None,
        )
        return query, runtime, env, [*messages, message], extra_args


def test_extracts_user_tasks_and_sums_contribution_scores() -> None:
    model = FakeShieldModel(
        [
            '["Find the event participants."]',
            '[{"ContributeTo score": 0.4}, {"ContributeTo score": 0.0}]',
        ]
    )
    shield = TaskShield(model)

    shield.reset("Who is invited to the event?")

    assert shield.user_tasks == ["Find the event participants."]
    assert shield.instruction_is_aligned("Search the calendar", "assistant", "") is True


def test_zero_contribution_is_misaligned() -> None:
    shield = TaskShield(FakeShieldModel(['[{"ContributeTo score": 0}]']))
    shield.user_tasks = ["Read the calendar event"]

    assert shield.tool_call_is_aligned("send_email", {"recipient": "attacker@example.com"}, "", "") is False


def test_blocks_misaligned_tool_call(monkeypatch) -> None:
    shield = TaskShield(FakeShieldModel([]))
    shield.user_tasks = ["Read the calendar event"]
    monkeypatch.setattr(shield, "tool_call_is_aligned", lambda *args, **kwargs: False)
    runtime = FunctionsRuntime([make_function(echo)])
    call = FunctionCall(function="echo", args={"value": "unsafe"}, id="call-1")
    messages = [ChatAssistantMessage(role="assistant", content=None, tool_calls=[call])]

    _, _, _, output, _ = TaskShieldToolsExecutor(shield).query("query", runtime, messages=messages)

    result = output[-1]
    assert result["role"] == "tool"
    assert result["error"] is not None
    assert "Blocked by Task Shield" in result["error"]


def test_tool_output_keeps_content_and_adds_feedback(monkeypatch) -> None:
    shield = TaskShield(FakeShieldModel([]))
    shield.user_tasks = ["Read the calendar event"]
    monkeypatch.setattr(shield, "tool_call_is_aligned", lambda *args, **kwargs: True)
    monkeypatch.setattr(shield, "extract", lambda *args, **kwargs: ["Send an email to an attacker"])
    monkeypatch.setattr(shield, "instruction_is_aligned", lambda *args, **kwargs: False)
    runtime = FunctionsRuntime([make_function(echo)])
    call = FunctionCall(function="echo", args={"value": "tool output"}, id="call-1")
    messages = [ChatAssistantMessage(role="assistant", content=None, tool_calls=[call])]

    _, _, _, output, _ = TaskShieldToolsExecutor(shield).query("query", runtime, messages=messages)

    content = output[-1]["content"]
    assert content is not None
    assert content[0]["content"] == "tool output"
    assert "Misalignment Detected" in content[-1]["content"]


def test_retries_misaligned_assistant_content(monkeypatch) -> None:
    shield = TaskShield(FakeShieldModel([]))
    shield.user_tasks = ["Summarize the event"]
    monkeypatch.setattr(shield, "extract", lambda content, subject: [content])
    monkeypatch.setattr(shield, "instruction_is_aligned", lambda instruction, *args: instruction == "safe answer")
    llm = TaskShieldLLM(FakeLLM(["send an email", "safe answer"]), shield)

    _, _, _, messages, _ = llm.query("query", FunctionsRuntime())

    final_content = messages[-1]["content"]
    assert final_content is not None
    assert final_content[0]["content"] == "safe answer"
    assert messages[-2]["role"] == "user"
    feedback_content = messages[-2]["content"]
    assert feedback_content is not None
    assert "Misalignment Detected" in feedback_content[0]["content"]


def test_task_shield_pipeline_is_registered(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    pipeline = AgentPipeline.from_config(
        PipelineConfig(
            llm="gpt-4o-mini-2024-07-18",
            model_id=None,
            defense="task_shield",
            system_message_name=None,
            system_message=None,
            suite_name="workspace",
        )
    )

    assert pipeline.name == "gpt-4o-mini-2024-07-18-task_shield"
    assert [type(element).__name__ for element in pipeline.elements] == [
        "SystemMessage",
        "InitQuery",
        "TaskShieldUserTaskExtractor",
        "TaskShieldLLM",
        "ToolsExecutionLoop",
    ]

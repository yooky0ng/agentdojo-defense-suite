from agentdojo.agent_pipeline import AgentPipeline, PipelineConfig
from agentdojo.defenses import progent_adapter
from agentdojo.functions_runtime import FunctionsRuntime, make_function


def echo(value: str) -> str:
    """Return a value.

    :param value: Value to return.
    """
    return value


class FakeSecagent:
    def __init__(self) -> None:
        self.available_tools = []
        self.generated_queries = []
        self.checked_calls = []
        self.allowed_tools = []

    def reset_security_policy(self, include_human_policy=False) -> None:
        assert include_human_policy is True

    def update_available_tools(self, tools) -> None:
        self.available_tools = tools

    def update_always_allowed_tools(self, tools, allow_all_no_arg_tools=False) -> None:
        self.allowed_tools = tools

    def generate_security_policy(self, query) -> None:
        self.generated_queries.append(query)

    def check_tool_call(self, name, args) -> None:
        self.checked_calls.append((name, args))

    def generate_update_security_policy(self, *args, **kwargs) -> None:
        raise AssertionError("Dynamic updates are disabled in this test")


def test_progent_is_registered(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    pipeline = AgentPipeline.from_config(
        PipelineConfig(
            llm="gpt-4o-mini-2024-07-18",
            model_id=None,
            defense="progent",
            suite_name="workspace",
            system_message_name=None,
            system_message=None,
        )
    )

    assert pipeline.name == "gpt-4o-mini-2024-07-18-progent"
    assert [type(element).__name__ for element in pipeline.elements] == [
        "SystemMessage",
        "InitQuery",
        "ProgentPolicyBootstrap",
        "OpenAILLM",
        "ToolsExecutionLoop",
    ]


def test_bootstrap_connects_upstream_policy_to_runtime(monkeypatch) -> None:
    fake_secagent = FakeSecagent()
    monkeypatch.setattr(progent_adapter, "_load_secagent", lambda: fake_secagent)
    runtime = FunctionsRuntime([make_function(echo)])
    component = progent_adapter.ProgentPolicyBootstrap("slack")

    component.query("repeat hello", runtime)
    result, error = runtime.run_function(None, "echo", {"value": "hello"})

    assert result == "hello"
    assert error is None
    assert fake_secagent.available_tools[0]["name"] == "echo"
    assert fake_secagent.generated_queries == ["repeat hello"]
    assert fake_secagent.checked_calls == [("echo", {"value": "hello"})]
    assert fake_secagent.allowed_tools == [
        "get_channels",
        "read_channel_messages",
        "read_inbox",
        "get_users_in_channel",
    ]


def test_loads_untouched_upstream_secagent() -> None:
    secagent = progent_adapter._load_secagent()
    assert secagent.__name__ == "agentdojo.defenses.progent.secagent"
    assert secagent.check_tool_call.__module__ == "agentdojo.defenses.progent.secagent.tool"

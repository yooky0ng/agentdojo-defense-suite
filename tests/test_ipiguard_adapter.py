from agentdojo.agent_pipeline import AgentPipeline, PipelineConfig
from agentdojo.defenses.ipiguard_adapter import _to_current_messages, _to_legacy_messages


def test_ipiguard_message_conversion_round_trip() -> None:
    messages = [{"role": "user", "content": [{"type": "text", "content": "hello"}]}]

    legacy_messages = _to_legacy_messages(messages)
    assert legacy_messages[0]["content"] == "hello"
    assert _to_current_messages(legacy_messages) == messages


def test_builds_pipeline_from_upstream_ipiguard(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    pipeline = AgentPipeline.from_config(
        PipelineConfig(
            llm="gpt-4o-mini-2024-07-18",
            model_id=None,
            defense="ipiguard",
            system_message_name=None,
            system_message=None,
        )
    )

    construct_llm = pipeline.elements[2]._element
    tools_loop = pipeline.elements[3]._element
    assert construct_llm.__class__.__module__ == "_agentdojo_upstream_ipiguard_llm"
    assert tools_loop.__class__.__module__ == "_agentdojo_upstream_ipiguard_tool_execution"
    assert tools_loop.executor.__class__.__module__ == "_agentdojo_upstream_ipiguard_tool_execution"
    assert tools_loop.executor.traverse_llm.__class__.__module__ == "_agentdojo_upstream_ipiguard_llm"

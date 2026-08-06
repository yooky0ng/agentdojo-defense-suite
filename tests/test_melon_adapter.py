from agentdojo.agent_pipeline import AgentPipeline, PipelineConfig
from agentdojo.defenses.melon_adapter import _to_current_messages, _to_legacy_messages


def test_melon_message_conversion_round_trip() -> None:
    messages = [{"role": "user", "content": [{"type": "text", "content": "hello"}]}]

    legacy_messages = _to_legacy_messages(messages)
    assert legacy_messages[0]["content"] == "hello"
    assert _to_current_messages(legacy_messages) == messages


def test_builds_detector_from_upstream_melon(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    pipeline = AgentPipeline.from_config(
        PipelineConfig(
            llm="gpt-4o-mini-2024-07-18",
            model_id=None,
            defense="melon",
            system_message_name=None,
            system_message=None,
        )
    )

    detector_adapter = pipeline.elements[-1].elements[-1]
    detector = detector_adapter._detector
    assert detector.__class__.__name__ == "MELON"
    assert detector.__class__.__module__ == "_agentdojo_upstream_melon_pi_detector"
    assert detector.query.__func__.__module__ == "_agentdojo_upstream_melon_pi_detector"
    assert detector.detect.__func__.__module__ == "_agentdojo_upstream_melon_pi_detector"
    assert detector.detection_model is pipeline.elements[2].client

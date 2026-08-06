from agentdojo.agent_pipeline import AgentPipeline, PipelineConfig


def test_builds_pipeline_from_upstream_drift(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    pipeline = AgentPipeline.from_config(
        PipelineConfig(
            llm="gpt-4o-mini-2024-07-18",
            model_id=None,
            defense="drift",
            system_message_name=None,
            system_message=None,
        )
    )

    assert pipeline.name == "gpt-4o-mini-2024-07-18-drift"
    assert [type(element).__name__ for element in pipeline.elements] == [
        "InitQuery",
        "DriftMessageAdapter",
        "DRIFTToolsExecutionLoop",
    ]
    upstream_llm = pipeline.elements[1]._drift_llm
    assert upstream_llm.__class__.__module__ == "DRIFTLLM"
    assert upstream_llm.args.dynamic_validation is True
    assert upstream_llm.args.build_constraints is True
    assert upstream_llm.args.injection_isolation is True

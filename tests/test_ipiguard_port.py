import hashlib
from unittest.mock import Mock

import networkx as nx

from agentdojo.agent_pipeline.agent_pipeline import DEFENSES, AgentPipeline, PipelineConfig
from agentdojo.agent_pipeline.ipiguard.ipiguard_llm import OpenAIConstructLLM, OpenAITraverseLLM
from agentdojo.agent_pipeline.ipiguard.tool_white_list import whitelist
from agentdojo.agent_pipeline.llms.openai_llm import OpenAILLM
from agentdojo.functions_runtime import FunctionCall


def test_ipiguard_is_registered() -> None:
    assert "ipiguard" in DEFENSES


def test_construct_dag_preserves_dependencies() -> None:
    llm = OpenAIConstructLLM(Mock(), "gpt-4o-mini-2024-07-18")
    dag = llm.construct_dag(
        '{"tool_calls": ['
        '{"id": "1", "function_name": "search_emails", "args": {"query": "x"}, "depends_on": []},'
        '{"id": "2", "function_name": "get_file_by_id", "args": {"file_id": "<unknown>: string"}, '
        '"depends_on": ["1"]}]}'
    )
    assert isinstance(dag, nx.DiGraph)
    assert list(nx.topological_sort(dag)) == ["1", "2"]
    assert dag.nodes["2"]["function_call"] == FunctionCall(
        id="2", function="get_file_by_id", args={"file_id": "<unknown>: string"}
    )


def test_upstream_whitelist_keeps_query_tools_and_excludes_action_tools() -> None:
    assert "search_emails" in whitelist
    assert "read_channel_messages" in whitelist
    assert "send_email" not in whitelist
    assert "send_money" not in whitelist


def test_upstream_prompts_are_preserved_verbatim() -> None:
    expected_hashes = {
        OpenAIConstructLLM._construct_dag_prompt: "16adb605b9e2d45c5e0a67b570961b7005ee0cf23c005ae35ea3ee3fbbcf96c1",
        OpenAITraverseLLM._args_update_prompt: "ffe71842903f1966b06c80c06aed34e991fee7cf683bc4fbf49ab7301cab0f83",
        OpenAITraverseLLM._tool_call_information: "2715855ca6fb42de83ee643d555169ffe14d903e89b597ff18fe75e90e9090f4",
        OpenAITraverseLLM._history_update_prompt: "925057ce54669079f0ea63d14e87993e8a9dd2241e2960861ed3a99d08f12335",
        OpenAITraverseLLM._history_expansion_prompt: "839c4fd8bea170e55257ad44d0d3c7fdb56812dfe3bdeda0bef777f1d11c1a5c",
        OpenAITraverseLLM._history_fix_prompt: "4eadf25d1ecb064770adb574ffabcd884c949d1d7d9f04e42a5f2aff0aa1ec7a",
    }
    for prompt, expected in expected_hashes.items():
        assert hashlib.sha256(prompt.encode()).hexdigest() == expected


def test_pipeline_builds_ipiguard_for_openai() -> None:
    base_llm = OpenAILLM(Mock(), "gpt-4o-mini-2024-07-18")
    pipeline = AgentPipeline.from_config(
        PipelineConfig(
            llm=base_llm,
            model_id=None,
            defense="ipiguard",
            system_message_name=None,
            system_message="system",
        )
    )
    assert pipeline.name == "None-ipiguard"
    assert len(list(pipeline.elements)) == 4

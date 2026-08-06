"""AgentDojo compatibility adapter for the upstream IPIGuard fork."""

import importlib.util
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

from agentdojo.agent_pipeline.base_pipeline_element import BasePipelineElement
from agentdojo.functions_runtime import EmptyEnv, Env, FunctionsRuntime
from agentdojo.types import ChatMessage, get_text_content_as_str, text_content_block_from_string


def _load_module(module_name: str, path: Path):
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load upstream IPIGuard module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _load_upstream_modules():
    root = Path(__file__).resolve().parent / "ipiguard" / "agentdojo" / "src" / "agentdojo"
    whitelist = _load_module(
        "_agentdojo_upstream_ipiguard_whitelist",
        root / "default_suites" / "v1" / "tools" / "tool_white_list.py",
    )
    upstream_openai = _load_module(
        "_agentdojo_upstream_ipiguard_openai_llm",
        root / "agent_pipeline" / "llms" / "openai_llm.py",
    )

    aliases = {
        "agentdojo.default_suites.v1.tools.tool_white_list": whitelist,
        "agentdojo.agent_pipeline.llms.openai_llm": upstream_openai,
    }
    previous = {name: sys.modules.get(name) for name in aliases}
    sys.modules.update(aliases)
    try:
        ipiguard_llm = _load_module(
            "_agentdojo_upstream_ipiguard_llm",
            root / "agent_pipeline" / "llms" / "ipiguard_llm.py",
        )
        tool_execution = _load_module(
            "_agentdojo_upstream_ipiguard_tool_execution",
            root / "agent_pipeline" / "tool_execution.py",
        )
    finally:
        for name, module in previous.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module
    return ipiguard_llm, tool_execution


def _to_legacy_messages(messages: Sequence[ChatMessage]) -> list[ChatMessage]:
    legacy_messages = []
    for message in messages:
        content = message.get("content")
        if isinstance(content, list):
            message = {**message, "content": get_text_content_as_str(content)}
        legacy_messages.append(message)
    return legacy_messages


def _to_current_messages(messages: Sequence[ChatMessage]) -> list[ChatMessage]:
    current_messages = []
    for message in messages:
        content = message.get("content")
        if isinstance(content, str):
            message = {**message, "content": [text_content_block_from_string(content)]}
        current_messages.append(message)
    return current_messages


class IPIGuardMessageAdapter(BasePipelineElement):
    """Adapt messages around one upstream IPIGuard pipeline element."""

    def __init__(self, element: BasePipelineElement) -> None:
        self._element = element

    def query(
        self,
        query: str,
        runtime: FunctionsRuntime,
        env: Env = EmptyEnv(),
        messages: Sequence[ChatMessage] = [],
        extra_args: dict = {},
    ) -> tuple[str, FunctionsRuntime, Env, Sequence[ChatMessage], dict]:
        query, runtime, env, messages, extra_args = self._element.query(
            query, runtime, env, _to_legacy_messages(messages), extra_args
        )
        return query, runtime, env, _to_current_messages(messages), extra_args


def build_ipiguard_elements(
    client,
    model: str,
    tool_output_formatter: Callable,
) -> tuple[BasePipelineElement, BasePipelineElement]:
    """Build the official IPIGuard construct and DAG execution elements."""
    ipiguard_llm, tool_execution = _load_upstream_modules()
    construct_llm = ipiguard_llm.OpenAIConstructLLM(client, model)
    traverse_llm = ipiguard_llm.OpenAITraverseLLM(client, model)
    executor = tool_execution.DagToolsExecutor(traverse_llm, tool_output_formatter)
    tools_loop = tool_execution.DagToolsExecutionLoop(executor)
    return IPIGuardMessageAdapter(construct_llm), IPIGuardMessageAdapter(tools_loop)

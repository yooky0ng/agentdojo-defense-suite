"""AgentDojo compatibility adapter for the upstream DRIFT repository."""

import logging
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from types import SimpleNamespace

from agentdojo.agent_pipeline.base_pipeline_element import BasePipelineElement
from agentdojo.functions_runtime import EmptyEnv, Env, FunctionsRuntime
from agentdojo.types import ChatMessage, text_content_block_from_string


def _add_drift_to_syspath() -> None:
    drift_root = Path(__file__).resolve().parent / "drift"
    drift_root_str = str(drift_root)
    if drift_root_str not in sys.path:
        sys.path.insert(0, drift_root_str)


class DriftMessageAdapter(BasePipelineElement):
    """Normalize upstream DRIFT's string content for current AgentDojo messages."""

    def __init__(self, drift_llm: BasePipelineElement) -> None:
        self._drift_llm = drift_llm
        self.name = getattr(drift_llm, "name", None)

    def query(
        self,
        query: str,
        runtime: FunctionsRuntime,
        env: Env = EmptyEnv(),
        messages: Sequence[ChatMessage] = [],
        extra_args: dict = {},
    ) -> tuple[str, FunctionsRuntime, Env, Sequence[ChatMessage], dict]:
        query, runtime, env, output_messages, extra_args = self._drift_llm.query(
            query, runtime, env, messages, extra_args
        )
        normalized_messages = []
        for message in output_messages:
            content = message.get("content")
            if isinstance(content, str):
                message = {**message, "content": [text_content_block_from_string(content)]}
            normalized_messages.append(message)
        return query, runtime, env, normalized_messages, extra_args


def build_drift_pipeline(model: str, tool_output_formatter: Callable):
    """Build the upstream DRIFT pipeline with all official defense stages enabled."""
    import agentdojo.task_suite  # noqa: F401
    from agentdojo.agent_pipeline.agent_pipeline import AgentPipeline
    from agentdojo.agent_pipeline.basic_elements import InitQuery
    from agentdojo.agent_pipeline.tool_execution import ToolsExecutor

    _add_drift_to_syspath()

    from client import GoogleModel, OpenAIModel, OpenRouterModel  # type: ignore[import-not-found]
    from DRIFTLLM import DRIFTLLM  # type: ignore[import-not-found]
    from DRIFTToolsExecutionLoop import DRIFTToolsExecutionLoop  # type: ignore[import-not-found]

    logger = logging.getLogger("agentdojo.drift")
    logger.setLevel(logging.INFO)

    if model.startswith(("gpt", "o1", "o3", "o4")):
        client = OpenAIModel(model=model, logger=logger)
    elif model.startswith("gemini"):
        client = GoogleModel(model=model, logger=logger)
    else:
        client = OpenRouterModel(model=model, logger=logger)

    args = SimpleNamespace(
        dynamic_validation=True,
        build_constraints=True,
        injection_isolation=True,
    )
    upstream_llm = DRIFTLLM(args, client=client, model=model, logger=logger)
    llm = DriftMessageAdapter(upstream_llm)
    tools_loop = DRIFTToolsExecutionLoop([ToolsExecutor(tool_output_formatter), llm])
    pipeline = AgentPipeline([InitQuery(), llm, tools_loop])
    pipeline.name = f"{model}-drift"
    return pipeline

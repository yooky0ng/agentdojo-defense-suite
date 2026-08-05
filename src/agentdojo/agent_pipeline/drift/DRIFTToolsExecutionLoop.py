"""DRIFT's call-error-aware tool execution loop."""

from collections.abc import Sequence

from agentdojo.agent_pipeline.base_pipeline_element import BasePipelineElement
from agentdojo.agent_pipeline.drift.compatibility import message_text
from agentdojo.agent_pipeline.drift.DRIFTLLM import DRIFTLLM
from agentdojo.functions_runtime import EmptyEnv, Env, FunctionsRuntime
from agentdojo.types import ChatMessage


class DRIFTToolsExecutionLoop(BasePipelineElement):
    """Execute tools and DRIFT until DRIFT returns no more tool calls."""

    def __init__(self, elements: Sequence[BasePipelineElement], max_iters: int = 15) -> None:
        self.elements = elements
        self.max_iters = max_iters

    def query(
        self,
        query: str,
        runtime: FunctionsRuntime,
        env: Env = EmptyEnv(),
        messages: Sequence[ChatMessage] = (),
        extra_args: dict = {},
    ) -> tuple[str, FunctionsRuntime, Env, Sequence[ChatMessage], dict]:
        for _ in range(self.max_iters):
            if not messages:
                break
            last_message = messages[-1]
            is_call_error = (
                last_message["role"] == "user"
                and "[CALL ERROR]" in message_text(last_message)
            )
            if not is_call_error and (
                last_message["role"] != "assistant" or not last_message.get("tool_calls")
            ):
                break
            for element in self.elements:
                if is_call_error and not isinstance(element, DRIFTLLM):
                    continue
                query, runtime, env, messages, extra_args = element.query(
                    query, runtime, env, messages, extra_args
                )
        return query, runtime, env, messages, extra_args

__all__ = ["DRIFTToolsExecutionLoop"]

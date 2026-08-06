"""AgentDojo compatibility adapter for the upstream MELON repository."""

import importlib.util
import sys
from collections.abc import Sequence
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType

from agentdojo.agent_pipeline.base_pipeline_element import BasePipelineElement
from agentdojo.functions_runtime import EmptyEnv, Env, FunctionsRuntime
from agentdojo.types import (
    ChatMessage,
    get_text_content_as_str,
    text_content_block_from_string,
)


@contextmanager
def _provide_unused_torch_import():
    """Satisfy upstream MELON's unused torch.nn.functional import if absent."""
    if importlib.util.find_spec("torch") is not None:
        yield
        return

    torch_module = ModuleType("torch")
    torch_module.__path__ = []
    nn_module = ModuleType("torch.nn")
    nn_module.__path__ = []
    functional_module = ModuleType("torch.nn.functional")
    torch_module.nn = nn_module
    nn_module.functional = functional_module
    temporary_modules = {
        "torch": torch_module,
        "torch.nn": nn_module,
        "torch.nn.functional": functional_module,
    }
    sys.modules.update(temporary_modules)
    try:
        yield
    finally:
        for module_name in temporary_modules:
            sys.modules.pop(module_name, None)


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


class MelonLLMAdapter(BasePipelineElement):
    """Let upstream MELON call a current AgentDojo LLM with legacy messages."""

    def __init__(self, llm: BasePipelineElement) -> None:
        self._llm = llm
        self.name = getattr(llm, "name", None)

    def query(
        self,
        query: str,
        runtime: FunctionsRuntime,
        env: Env = EmptyEnv(),
        messages: Sequence[ChatMessage] = [],
        extra_args: dict = {},
    ) -> tuple[str, FunctionsRuntime, Env, Sequence[ChatMessage], dict]:
        query, runtime, env, output_messages, extra_args = self._llm.query(
            query,
            runtime,
            env,
            _to_current_messages(messages),
            extra_args,
        )
        return query, runtime, env, _to_legacy_messages(output_messages), extra_args


class MelonMessageAdapter(BasePipelineElement):
    """Adapt current AgentDojo messages around the upstream MELON detector."""

    def __init__(self, detector: BasePipelineElement) -> None:
        self._detector = detector

    def query(
        self,
        query: str,
        runtime: FunctionsRuntime,
        env: Env = EmptyEnv(),
        messages: Sequence[ChatMessage] = [],
        extra_args: dict = {},
    ) -> tuple[str, FunctionsRuntime, Env, Sequence[ChatMessage], dict]:
        with _provide_unused_torch_import():
            query, runtime, env, output_messages, extra_args = self._detector.query(
                query,
                runtime,
                env,
                _to_legacy_messages(messages),
                extra_args,
            )
        return query, runtime, env, _to_current_messages(output_messages), extra_args


def _load_upstream_melon_class():
    module_name = "_agentdojo_upstream_melon_pi_detector"
    if module_name in sys.modules:
        return sys.modules[module_name].MELON

    source_path = Path(__file__).resolve().parent / "melon" / "pi_detector.py"
    spec = importlib.util.spec_from_file_location(module_name, source_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load upstream MELON from {source_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module.MELON


def build_melon_detector(llm: BasePipelineElement, threshold: float = 0.1) -> BasePipelineElement:
    """Build upstream MELON while keeping compatibility outside its source."""
    client = getattr(llm, "client", None)
    if client is None:
        raise ValueError("MELON requires an OpenAI LLM client")

    melon_class = _load_upstream_melon_class()
    detector = melon_class.__new__(melon_class)
    detector_base = melon_class.__mro__[1]
    detector_base.__init__(detector, mode="full_conversation", raise_on_injection=False)
    detector.llm = MelonLLMAdapter(llm)
    detector.threshold = threshold
    detector.detection_model = client
    return MelonMessageAdapter(detector)

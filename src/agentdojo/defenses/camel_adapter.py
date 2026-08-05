"""AgentDojo compatibility adapter for the upstream CaMeL repository."""

import sys
from pathlib import Path
from typing import cast

from pydantic_ai.models import KnownModelName

from agentdojo.models import MODEL_PROVIDERS, ModelsEnum


def _add_camel_to_syspath() -> None:
    camel_src = Path(__file__).resolve().parent / "camel" / "src"
    camel_src_str = str(camel_src)
    if camel_src_str not in sys.path:
        sys.path.insert(0, camel_src_str)


def build_camel_pipeline(model: str, suite_name: str):
    """Build the upstream CaMeL pipeline with its default configuration."""
    model_enum = ModelsEnum(model)
    provider = MODEL_PROVIDERS[model_enum]
    if provider not in {"openai", "anthropic", "google"}:
        raise ValueError(f"CaMeL does not support the AgentDojo provider '{provider}'")

    # Finish AgentDojo suite registration before CaMeL imports its environment
    # classes. This avoids a circular import on AgentDojo 0.1.35.
    import agentdojo.task_suite  # noqa: F401

    _add_camel_to_syspath()

    from camel.interpreter.interpreter import MetadataEvalMode  # type: ignore[import-not-found]
    from camel.models import make_tools_pipeline  # type: ignore[import-not-found]

    return make_tools_pipeline(
        model=cast(KnownModelName, f"{provider}:{model}"),
        use_original=False,
        replay_with_policies=False,
        attack_name="important_instructions",
        reasoning_effort="medium",
        thinking_budget_tokens=None,
        suite=suite_name,
        ad_defense=None,
        eval_mode=MetadataEvalMode.NORMAL,
        q_llm=None,
    )

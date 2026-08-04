"""DRIFT's call-error-aware tool execution loop."""

from agentdojo.agent_pipeline.drift_simplified import DRIFTToolsExecutionLoop as _DRIFTToolsExecutionLoop


class DRIFTToolsExecutionLoop(_DRIFTToolsExecutionLoop):
    """Current-AgentDojo adapter for upstream DRIFT's retry loop."""

__all__ = ["DRIFTToolsExecutionLoop"]

# app/pipeline/modes.py

"""
Lightweight worker-pipeline mode definitions.

The worker pipeline mode is an internal orchestration concept, not an HTTP API concept. This module is intentionally tiny and dependency-free so it is safe to import from both web-safe enqueue code and worker-side orchestrator code.

modes:
    default  Normal / incremental pipeline run.
    repair   Re-run only missing/failed/skipped stages (planned, not yet implemented).
    redo     Force a full re-run (planned, not yet implemented).
"""

from enum import Enum


class PipelineMode(str, Enum):
    """
    Internal worker pipeline mode.

    Deliberately separate from the public HTTP ingestion mode: routes accept integer modes (0..3) and convert them to PipelineMode before enqueueing.
    """

    DEFAULT = "default"
    REPAIR  = "repair"
    REDO    = "redo"


def parse_pipeline_mode(value: str | PipelineMode) -> PipelineMode:
    """
    Normalize a worker mode value into a PipelineMode enum member.

    Accepts both PipelineMode enum values and their canonical string equivalents.
    RQ can serialize enums directly, but normalizing at the enqueue boundary
    keeps downstream code simpler and guards against unexpected string values.

    Raises:
        ValueError: If `value` is a string that does not match any known mode.
    """
    if isinstance(value, PipelineMode):
        return value

    match value:
        case "default": return PipelineMode.DEFAULT
        case "repair":  return PipelineMode.REPAIR
        case "redo":    return PipelineMode.REDO

    raise ValueError(f"Invalid worker pipeline mode: {value!r}")



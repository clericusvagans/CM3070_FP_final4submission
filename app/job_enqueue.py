# app/job_enqueue.py

"""
Web-safe RQ enqueue helpers.

FastAPI imports routes at startup. If routes import a module that imports the orchestrator or OCR/ML code, the web process may load heavy dependencies or trigger import-time side effects.

This module must remain web-safe:
    - No imports of pipeline orchestrators or OCR / ML code
    - No DB reads/writes
    - Only queue setup + enqueue by dotted string reference

IMPORANT:
- Enqueueing via dotted string path avoids importing the worker callable in the web process and its import-time side effects:
    q.enqueue("app.pipeline.orchestrator_std.process_document", doc_id)

- Mode boundary: The public HTTP API uses integer modes (0..3); the worker pipeline uses PipelineMode enums. Conversion happens here, at the enqueue boundary.
"""

from redis import Redis
from rq import Queue

from app.config.config import settings
from app.pipeline.modes import PipelineMode, parse_pipeline_mode
from app.utils.logger import logger


redis = Redis.from_url(settings.REDIS_URL)
q = Queue(
    settings.RQ_QUEUE_NAME,
    connection=redis,
    default_timeout=settings.RQ_JOB_TIMEOUT_SECONDS,
)


def run_pipeline(
    doc_id: int,
    *,
    mode: PipelineMode | str = PipelineMode.DEFAULT,
    stages: list[str] | None = None,
):
    """
    Enqueue the standard (DB-backed) document pipeline for a single document.

    Enqueues by string path so FastAPI does not import heavy pipeline code.
    Does not touch the DB.

    Args:
        doc_id:  Target document ID.
        mode:    Worker pipeline mode. Accepts PipelineMode or its canonical
                 string (`"default"`, `"repair"`, `"redo"`).
        stages:  Optional future stage filter (accepted for API compatibility;
                 stage-aware execution is not yet implemented).

    Returns:
        The enqueued `rq.job.Job` object.
    """
    mode = parse_pipeline_mode(mode)

    logger.info(
        f"[Enqueue] Scheduling pipeline doc_id={doc_id} mode={mode.value} stages={stages}"
    )

    return q.enqueue(
        "app.pipeline.orchestrator_std.process_document",
        doc_id,
        mode=mode.value,
        stages=stages,
    )


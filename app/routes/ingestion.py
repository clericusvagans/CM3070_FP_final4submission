# app/routes/ingestion.py

"""
FastAPI ingestion endpoints.

Strictly for route handling and task enqueueing. Must remain web-safe: no pipeline/OCR/ML imports, no direct DB writes beyond delegating to `ingest_file_async`.
"""

from fastapi import Response, APIRouter, UploadFile, File, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import get_session
from app.ingestion.ingest_file import ingest_file_async
from app.job_enqueue import run_pipeline
from app.pipeline.modes import PipelineMode
from app.utils.logger import logger

router = APIRouter(tags=["ingestion"])


@router.post("/ingest/file")
async def ingest_single_file(
    response: Response,
    mode: int = Query(
        1, ge=0, le=3,
        description=(
            "HTTP ingestion mode: "
            "0=always create a new document row, "
            "1=dedup-aware default/incremental, "
            "2=dedup-aware repair, "
            "3=dedup-aware redo-all"
        ),
    ),
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_session),
):
    """
    Ingest a single file and schedule pipeline processing.

    HTTP mode semantics
    -------------------
    mode=0  Always create a new Document row and enqueue a full pipeline run,
            even if the same bytes already exist.

    mode=1  Duplicate-aware / default-incremental.
            new hash → create Document + enqueue default worker run.
            duplicate → reuse canonical Document, do not enqueue.

    mode=2  Duplicate-aware / repair.
            new hash → create Document + enqueue default worker run.
            duplicate → reuse canonical Document + enqueue repair worker run.

    mode=3  Duplicate-aware / redo-all.
            new hash → create Document + enqueue default worker run.
            duplicate → reuse canonical Document + enqueue redo worker run.

    The ingestion layer performs all DB writes and commits. This route only:
        - calls ingestion
        - exposes outcome signals as response headers
        - converts the HTTP mode to the internal PipelineMode before enqueueing
    """
    result = await ingest_file_async(db, file, mode=mode)

    response.headers["X-Ingest-Outcome"] = result.outcome
    response.headers["X-Content-Hash"] = f"{result.hash_algo}:{result.content_hash}"
    response.headers["X-Duplicate-Group-Id"] = str(result.duplicate_group_id)

    # Enqueue policy:
    # Any newly-created Document (outcomes: created / duplicate_created / repaired)
    # always gets a default pipeline run. For a true "duplicate" outcome the HTTP
    # mode decides whether and how to enqueue.

    scheduled = False
    job_id    = None

    if result.outcome in ("created", "duplicate_created", "repaired"):
        job = run_pipeline(result.doc_id, mode=PipelineMode.DEFAULT)
        scheduled = True
        job_id = getattr(job, "id", None)
        logger.info(
            f"[Ingestion] Pipeline scheduled doc_id={result.doc_id} job_id={job_id} "
            f"outcome={result.outcome} http_mode={mode} worker_mode={PipelineMode.DEFAULT.value}"
        )

    elif result.outcome == "duplicate":
        if mode == 1:
            logger.info(
                f"[Ingestion] Duplicate detected; not scheduling pipeline "
                f"doc_id={result.doc_id} http_mode={mode}"
            )

        elif mode == 2:
            job = run_pipeline(result.doc_id, mode=PipelineMode.REPAIR)
            scheduled = True
            job_id = getattr(job, "id", None)
            logger.info(
                f"[Ingestion] Duplicate; scheduled repair "
                f"doc_id={result.doc_id} job_id={job_id} http_mode={mode}"
            )

        elif mode == 3:
            job = run_pipeline(result.doc_id, mode=PipelineMode.REDO)
            scheduled = True
            job_id = getattr(job, "id", None)
            logger.info(
                f"[Ingestion] Duplicate; scheduled redo "
                f"doc_id={result.doc_id} job_id={job_id} http_mode={mode}"
            )

        else:
            # mode=0 always creates a new row, so ingest_core returns
            # "duplicate_created", not "duplicate".
            logger.warning(
                f"[Ingestion] Unexpected outcome/mode combination "
                f"outcome={result.outcome} http_mode={mode} doc_id={result.doc_id}; "
                f"not scheduling pipeline"
            )

    else:
        logger.error(
            f"[Ingestion] Unrecognized outcome={result.outcome!r} "
            f"doc_id={result.doc_id} http_mode={mode}; not scheduling pipeline"
        )

    return {
        "document_id": result.doc_id,
        "status": "ingesting" if scheduled else "duplicate",
        "outcome": result.outcome,
        "job_id": job_id,
        "mode": mode,
        "duplicate_group_id": result.duplicate_group_id,
        "canonical_doc_id": result.canonical_doc_id,
    }



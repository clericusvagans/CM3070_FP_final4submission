# app/utils/job_tracking.py

from datetime import datetime, timezone
from typing import Any, Optional

from sqlalchemy.orm import Session

from app.models import Job
from app.utils.logger import logger


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_job_row(
    db: Session,
    *,
    job_id: str,
    document_id: int,
    stage: str,
    status: str,
) -> Job:
    """
    Ensure a Job row exists for the given RQ job ID.
    Creates the row if absent; otherwise updates stage and status.
    """
    job = db.get(Job, job_id)
    if job is None:
        job = Job(
            id=job_id,
            document_id=document_id,
            stage=stage,
            status=status,
        )
        db.add(job)
    else:
        job.stage  = stage
        job.status = status

    db.commit()
    db.refresh(job)
    return job


def append_event(
    db: Session,
    *,
    job_id: str,
    level: str,
    stage: str,
    message: str,
    data: Optional[dict[str, Any]] = None,
) -> None:
    """
    Append an event dict into Job.log["events"] (JSONB).
    Uses copy-on-write to trigger SQLAlchemy's change detection.
    Best-effort: silently skips if the job row does not exist.
    """
    job = db.get(Job, job_id)
    if job is None:
        logger.warning(f"[append_event] Job {job_id} missing")
        return

    log = dict(job.log or {})
    events = list(log.get("events") or [])
    events.append({
        "ts":      _utc_now_iso(),
        "level":   level,
        "stage":   stage,
        "message": message,
        "data":    data or {},
    })
    log["events"] = events
    job.log = log
    db.commit()


def set_status(db: Session, *, job_id: str, status: str) -> None:
    """Update Job.status. Best-effort: silently skips if the job row is missing."""
    job = db.get(Job, job_id)
    if job is None:
        logger.warning(f"[set_status] Job {job_id} missing")
        return
    job.status = status
    db.commit()


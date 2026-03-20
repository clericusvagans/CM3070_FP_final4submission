# app/ingestion/ingest_file.py

"""
FastAPI ingestion adapter.

Adapts HTTP file uploads (UploadFile) to the transport-agnostic core ingestion logic in `ingest_core`.
"""

import asyncio
import tempfile
from pathlib import Path

from fastapi import UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from app.ingestion.ingest_core import IngestResult, ingest_raw_path_async


async def _write_tempfile_offloop(data: bytes) -> Path:
    """
    Write upload bytes to a temporary file without blocking the event loop.

    `ingest_core` requires a filesystem path (`src_path`) so it can copy
    the file into canonical SOURCEFILE storage. Writing temp files is blocking
    disk I/O, therefore `asyncio.to_thread()`is used.

    `delete=False` is required so the path persists after the file is closed
    and can be copied by `ingest_core`.

    Returns:
        Path to the created temporary file.
    """
    def _blocking_write() -> str:
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(data)
            return tmp.name

    return Path(await asyncio.to_thread(_blocking_write))


async def _unlink_offloop(path: Path) -> None:
    """Temp-file deletion off the event loop (best-effort)"""
    def _blocking_unlink() -> None:
        try:
            path.unlink(missing_ok=True)
        except Exception:
            pass

    await asyncio.to_thread(_blocking_unlink)


async def ingest_file_async(
    db: AsyncSession,
    file: UploadFile,
    *,
    mode: int = 1,
) -> IngestResult:
    """
    Async adapter that ingests a FastAPI UploadFile.

    Workflow:
        1. Read upload bytes via `await file.read()`.
        2. Write a temp file off the event loop (blocking disk I/O).
        3. Delegate to `ingest_raw_path_async()` for canonical ingestion and commit.
        4. Always clean up the temp file (best-effort).

    Note: It reads the entire upload into memory. All DB commits happen inside
    `ingest_raw_path_async()`; the route must not commit.

    Args:
        db:   Request-scoped AsyncSession.
        file: FastAPI UploadFile.
        mode: Public HTTP ingestion mode (0=always create, 1=dedup default,
              2=dedup repair, 3=dedup redo).

    Returns:
        IngestResult (doc_id, outcome, hash_algo, content_hash,
        duplicate_group_id, canonical_doc_id).

    Raises:
        ValueError: If the upload is empty.
    """
    original_name = Path(file.filename or "upload.bin").name

    data = await file.read()
    if not data:
        raise ValueError("Empty upload")

    tmp_path = await _write_tempfile_offloop(data)
    try:
        return await ingest_raw_path_async(
            db,
            src_path=tmp_path,
            original_filename=original_name,
            mode=mode,
        )
    finally:
        await _unlink_offloop(tmp_path)



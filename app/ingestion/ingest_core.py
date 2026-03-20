# app/ingestion/ingest_core.py

"""
Core ingestion logic (transport-agnostic).

Used by all possible ingestion entry points (API uploads, batch folder ingestion,
migrations, etc.).

Overview:
---------
Given a local filesystem path and the original client filename, ingestion:

    1) Validates the file type using the original filename extension.
    2) Computes a stable sha256 content hash and byte size.
    3) Resolves or creates a DuplicateGroup for that content hash using
       INSERT ... ON CONFLICT DO NOTHING + SELECT ... FOR UPDATE, which
       avoids exception-driven race handling (no IntegrityError catch/retry,
       no rollback at the helper level) and serializes concurrent same-hash
       decisions through a single locked row.
    4) Depending on the HTTP ingestion mode, either reuses the canonical
       existing Document or creates a fresh one linked to the same group.
    5) Persists the raw artifact under a doc-id-anchored layout:
           <OUTPUT_ROOT>/documents/<doc_id:09d>/SOURCEFILE/<original_filename>
    6) Returns an IngestResult describing the outcome.

Outcomes
--------
    created            New Document created; no prior Document existed for this hash.
    duplicate          Document already existed and SOURCEFILE was present (no-op).
    repaired           Document existed but SOURCEFILE was missing; re-copied.
    duplicate_created  New Document created even though the same bytes already
                       existed (mode=0 "always create" behaviour).

HTTP mode semantics
-------------------
    mode=0  Always create a new Document row and enqueue a full pipeline run.
    mode=1  Dedup-aware / default-incremental. New → create. Duplicate → reuse.
    mode=2  Dedup-aware / repair. New → create. Duplicate → reuse.
    mode=3  Dedup-aware / redo-all. New → create. Duplicate → reuse.

For modes 2 and 3 the difference in worker behaviour (REPAIR vs REDO) is
applied by the route layer after ingestion returns — not here.

Key schema constraint
---------------------
Document.source_uri is NOT NULL, but the desired storage layout is
doc-id-anchored and the ID is not known until after INSERT. Therefore ingestion
inserts a Document with a clearly-invalid placeholder URI, flushes to allocate
doc.id, copies the file to its final location, then updates source_uri and
commits.
"""

import asyncio
import hashlib
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from uuid import uuid4

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Session
from sqlalchemy.dialects.postgresql import insert

from app.models import DocStatus, Document, DuplicateGroup
from app.utils.logger import logger
from app.utils.storage import SUPPORTED_EXTS, resolve_local_path, sourcefile_dir, to_output_uri

DEFAULT_HASH_ALGO = "sha256"

IngestOutcome = Literal["created", "duplicate", "repaired", "duplicate_created"]


@dataclass(frozen=True)
class IngestResult:
    """
    Result envelope returned by all ingestion functions.

    Attributes:
        doc_id:             Stable DB identifier for downstream scheduling.
        outcome:            One of: created | duplicate | repaired | duplicate_created.
        hash_algo:          Hash algorithm used (always "sha256").
        content_hash:       Hex digest of the file content.
        duplicate_group_id: ID of the DuplicateGroup row for this content hash.
        canonical_doc_id:   ID of the canonical (lowest-id) Document in the group.
                            Equals doc_id when this document itself is canonical.
    """
    doc_id: int
    outcome: IngestOutcome
    hash_algo: str
    content_hash: str
    duplicate_group_id: int
    canonical_doc_id: int


# ─── Helpers ───

def compute_sha256_and_size(
        path: Path,
        chunk_size: int = 8 * 1024 * 1024,
) -> tuple[str, int]:
    """
    Compute a SHA-256 digest and byte size for a file using streaming reads.

    Avoids loading large documents into memory. Returns a lowercase hex digest
    suitable for a fixed-length (64-char) DB column.
    """
    h = hashlib.sha256()
    total = 0
    with path.open("rb") as f:
        while chunk := f.read(chunk_size):
            total += len(chunk)
            h.update(chunk)
    return h.hexdigest(), total


def _copy_raw_into_doc(src_path: Path, doc_id: int, original_filename: str) -> Path:
    """
    Copy the raw upload into the canonical SOURCEFILE directory for a document.

    Destination: `<OUTPUT_ROOT>/documents/<doc_id:09d>/SOURCEFILE/<filename>`

    If a file with the same name already exists (e.g. during a repair of a
    previously ingested document) a UUID prefix is prepended to avoid silently
    overwriting it.
    """
    raw_dir = sourcefile_dir(doc_id)
    raw_dir.mkdir(parents=True, exist_ok=True)

    dst = raw_dir / original_filename
    if dst.exists():
        dst = raw_dir / f"{uuid4().hex}_{original_filename}"

    shutil.copyfile(src_path, dst)
    return dst


def _set_ingestion_metadata(
    doc: Document,
    *,
    ingestion_mode: int,
    duplicate_group_id: int,
    duplicate_status: str,
    canonical_doc_id: int,
) -> None:
    """
    Write ingestion provenance into doc.doc_metadata.

    Fields written
    --------------
    ingestion_mode:           Public HTTP mode used when this document was created.
    duplicate_group_id:       Copied into metadata for inspection without a join.
    duplicate_status:         `"unique"` | `"canonical"` | `"member"`.
    duplicate_of_document_id: Canonical doc ID, or None when this doc is canonical.
    """
    meta = dict(doc.doc_metadata or {})
    meta["ingestion_mode"] = ingestion_mode
    meta["duplicate_group_id"] = duplicate_group_id
    meta["duplicate_status"] = duplicate_status
    meta["duplicate_of_document_id"] = canonical_doc_id if canonical_doc_id != doc.id else None
    doc.doc_metadata = meta


def _select_canonical_doc_sync(db: Session, group_id: int) -> Document | None:
    """
    Return the canonical Document for a group (lowest id), or None (sync).
    Current policy: the document with the smallest ID is canonical; stable once set.
    """
    return (
        db.query(Document)
        .filter(Document.duplicate_group_id == group_id)
        .order_by(Document.id.asc())
        .first()
    )


async def _select_canonical_doc_async(db: AsyncSession, group_id: int) -> Document | None:
    """Async equivalent of `_select_canonical_doc_sync()`."""
    res = await db.execute(
        select(Document)
        .where(Document.duplicate_group_id == group_id)
        .order_by(Document.id.asc())
        .limit(1)
    )
    return res.scalar_one_or_none()


def _lock_or_create_duplicate_group_sync(
    db: Session,
    *,
    hash_algo: str,
    content_hash: str,
) -> DuplicateGroup:
    """
    Get or create the DuplicateGroup for a content hash, then lock it (sync).

    Uses INSERT ... ON CONFLICT DO NOTHING followed by SELECT ... FOR UPDATE.
    This avoids exception-driven race handling and serializes concurrent same-hash
    decisions through a single locked row.
    """
    db.execute(
        insert(DuplicateGroup)
        .values(hash_algo=hash_algo, content_hash=content_hash)
        .on_conflict_do_nothing(index_elements=["hash_algo", "content_hash"])
    )
    return (
        db.query(DuplicateGroup)
        .filter(
            DuplicateGroup.hash_algo == hash_algo,
            DuplicateGroup.content_hash == content_hash,
        )
        .with_for_update()
        .one()
    )


async def _lock_or_create_duplicate_group_async(
    db: AsyncSession,
    *,
    hash_algo: str,
    content_hash: str,
) -> DuplicateGroup:
    """Async equivalent of `_lock_or_create_duplicate_group_sync()`."""
    await db.execute(
        insert(DuplicateGroup)
        .values(hash_algo=hash_algo, content_hash=content_hash)
        .on_conflict_do_nothing(index_elements=["hash_algo", "content_hash"])
    )
    res = await db.execute(
        select(DuplicateGroup)
        .where(
            DuplicateGroup.hash_algo == hash_algo,
            DuplicateGroup.content_hash == content_hash,
        )
        .with_for_update()
    )
    return res.scalar_one()


# ──── Public API ────

def ingest_raw_path_sync(
    db: Session,
    *,
    src_path: Path,
    original_filename: str,
    mode: int = 1,
) -> IngestResult:
    """
    Ingest a raw document from an existing filesystem path (sync Session).

    Steps:
        1. Validate file type via original_filename extension (not src_path.suffix —
           API uploads are often suffix-less temp files).
        2. Compute sha256 hash + byte size.
        3. Lock or create DuplicateGroup (concurrency-safe).
        4. If mode 1/2/3 and a canonical Document already exists:
              SOURCEFILE present  → return outcome="duplicate" (no-op).
              SOURCEFILE missing  → re-copy and return outcome="repaired".
        5. Otherwise create a new Document:
              Insert placeholder source_uri (NOT NULL workaround), flush to
              allocate doc.id, copy SOURCEFILE, update source_uri, commit.
              Return outcome="created" (first time) or "duplicate_created" (mode=0).

    Args:
        db:                SQLAlchemy Session (sync).
        src_path:          Existing local file path (often a temp file).
        original_filename: Client filename used for validation and SOURCEFILE naming.
        mode:              Public HTTP ingestion mode (0-3).

    Returns:
        IngestResult.

    Raises:
        FileNotFoundError, ValueError, OSError, SQLAlchemyError.
    """
    if mode not in (0, 1, 2, 3):
        raise ValueError("mode must be 0, 1, 2, or 3")
    if not src_path.exists():
        raise FileNotFoundError(src_path)

    ext = Path(original_filename).suffix.lower()
    if ext not in SUPPORTED_EXTS:
        raise ValueError(f"Unsupported file type: {ext}")

    hash_algo = DEFAULT_HASH_ALGO
    content_hash, byte_size = compute_sha256_and_size(src_path)

    group = _lock_or_create_duplicate_group_sync(db, hash_algo=hash_algo, content_hash=content_hash)
    existing = _select_canonical_doc_sync(db, group.id)

    # Fast path: existing canonical document
    if existing is not None and mode in (1, 2, 3):
        try:
            if existing.source_uri and resolve_local_path(existing.source_uri).exists():
                logger.info(
                    f"[Ingestion] Duplicate detected; reusing canonical doc_id={existing.id} "
                    f"group_id={group.id} mode={mode}"
                )
                return IngestResult(
                    doc_id=existing.id, outcome="duplicate",
                    hash_algo=hash_algo, content_hash=content_hash,
                    duplicate_group_id=group.id, canonical_doc_id=existing.id,
                )
        except Exception:
            logger.warning(
                f"[Ingestion] Duplicate exists but SOURCEFILE is unreadable; repairing "
                f"doc_id={existing.id} group_id={group.id}"
            )

        # SOURCEFILE missing — re-copy into the existing doc's directory.
        logger.warning(
            f"[Ingestion] Missing SOURCEFILE; re-copying "
            f"doc_id={existing.id} group_id={group.id}"
        )
        dst = _copy_raw_into_doc(src_path, existing.id, original_filename)
        existing.source_uri = to_output_uri(dst)
        existing.byte_size = byte_size
        _set_ingestion_metadata(
            existing, ingestion_mode=mode,
            duplicate_group_id=group.id, duplicate_status="canonical",
            canonical_doc_id=existing.id,
        )
        db.commit()
        logger.info(f"[Ingestion] Repair complete: SOURCEFILE re-copied to {dst} for doc_id={existing.id}")
        return IngestResult(
            doc_id=existing.id, outcome="repaired",
            hash_algo=hash_algo, content_hash=content_hash,
            duplicate_group_id=group.id, canonical_doc_id=existing.id,
        )

    # Create new Document
    # A clearly-invalid placeholder satisfies the NOT NULL constraint on
    # source_uri while a flush is executed to allocate doc.id.
    placeholder_uri  = f"pending://raw/{uuid4().hex}"
    dst: Path | None = None
    canonical_doc_id = existing.id if existing is not None else None

    doc = Document(
        duplicate_group_id=group.id,
        filename=original_filename,
        source_uri=placeholder_uri,
        status=DocStatus.queued,
        hash_algo=hash_algo,
        content_hash=content_hash,
        byte_size=byte_size,
    )

    try:
        db.add(doc)
        db.flush()   # allocates doc.id

        if canonical_doc_id is None:
            canonical_doc_id = doc.id

        duplicate_status = "unique" if existing is None else "member"
        _set_ingestion_metadata(
            doc, ingestion_mode=mode,
            duplicate_group_id=group.id, duplicate_status=duplicate_status,
            canonical_doc_id=canonical_doc_id,
        )

        dst = _copy_raw_into_doc(src_path, doc.id, original_filename)
        doc.source_uri = to_output_uri(dst)
        db.commit()

        outcome: IngestOutcome = "created" if existing is None else "duplicate_created"
        logger.info(
            f"[Ingestion] Created document doc_id={doc.id} group_id={group.id} "
            f"canonical_doc_id={canonical_doc_id} outcome={outcome} mode={mode}"
        )
        return IngestResult(
            doc_id=doc.id, outcome=outcome,
            hash_algo=hash_algo, content_hash=content_hash,
            duplicate_group_id=group.id, canonical_doc_id=canonical_doc_id,
        )

    except Exception:
        # Roll back the placeholder row so no pending:// URIs persist in the DB.
        try:
            db.rollback()
        except Exception as rb_err:
            logger.error(f"[Ingestion] Rollback failed: {rb_err}")

        # Cleanup of any partially copied SOURCEFILE (best effort).
        if dst is not None:
            try:
                if dst.exists():
                    dst.unlink()
            except Exception as fs_err:
                logger.warning(f"[Ingestion] Failed to clean up partial SOURCEFILE {dst}: {fs_err}")
        raise


async def ingest_raw_path_async(
    db: AsyncSession,
    *,
    src_path: Path,
    original_filename: str,
    mode: int = 1,
) -> IngestResult:
    """
    Async counterpart to `ingest_raw_path_sync()`. Preserves identical
    semantics and outcomes.

    Blocking filesystem operations (hash computation, file copy) are executed
    in a worker thread via `asyncio.to_thread()` to avoid blocking the event loop.

    Args:
        db:                SQLAlchemy AsyncSession.
        src_path:          Existing local file path.
        original_filename: Client filename used for validation and SOURCEFILE naming.
        mode:              Public HTTP ingestion mode (0-3).

    Returns:
        IngestResult.

    Raises:
        FileNotFoundError, ValueError, OSError, SQLAlchemyError.
    """
    if mode not in (0, 1, 2, 3):
        raise ValueError("mode must be 0, 1, 2, or 3")
    if not src_path.exists():
        raise FileNotFoundError(src_path)

    ext = Path(original_filename).suffix.lower()
    if ext not in SUPPORTED_EXTS:
        raise ValueError(f"Unsupported file type: {ext}")

    hash_algo    = DEFAULT_HASH_ALGO
    content_hash, byte_size = await asyncio.to_thread(compute_sha256_and_size, src_path)

    group    = await _lock_or_create_duplicate_group_async(db, hash_algo=hash_algo, content_hash=content_hash)
    existing = await _select_canonical_doc_async(db, group.id)

    # Fast path: existing canonical document
    if existing is not None and mode in (1, 2, 3):
        try:
            if existing.source_uri and resolve_local_path(existing.source_uri).exists():
                logger.info(
                    f"[Ingestion] Duplicate detected; reusing canonical doc_id={existing.id} "
                    f"group_id={group.id} mode={mode}"
                )
                return IngestResult(
                    doc_id=existing.id, outcome="duplicate",
                    hash_algo=hash_algo, content_hash=content_hash,
                    duplicate_group_id=group.id, canonical_doc_id=existing.id,
                )
        except Exception:
            logger.warning(
                f"[Ingestion] Duplicate exists but SOURCEFILE is unreadable; repairing "
                f"doc_id={existing.id} group_id={group.id}"
            )

        logger.warning(
            f"[Ingestion] Missing SOURCEFILE; re-copying "
            f"doc_id={existing.id} group_id={group.id}"
        )
        dst = await asyncio.to_thread(
            lambda: _copy_raw_into_doc(src_path, existing.id, original_filename)
        )
        existing.source_uri = to_output_uri(dst)
        existing.byte_size  = byte_size
        _set_ingestion_metadata(
            existing, ingestion_mode=mode,
            duplicate_group_id=group.id, duplicate_status="canonical",
            canonical_doc_id=existing.id,
        )
        await db.commit()
        logger.info(f"[Ingestion] Repair complete: SOURCEFILE re-copied to {dst} for doc_id={existing.id}")
        return IngestResult(
            doc_id=existing.id, outcome="repaired",
            hash_algo=hash_algo, content_hash=content_hash,
            duplicate_group_id=group.id, canonical_doc_id=existing.id,
        )

    # Create new Document
    placeholder_uri = f"pending://raw/{uuid4().hex}"
    dst: Path | None = None
    canonical_doc_id = existing.id if existing is not None else None

    doc = Document(
        duplicate_group_id=group.id,
        filename=original_filename,
        source_uri=placeholder_uri,
        status=DocStatus.queued,
        hash_algo=hash_algo,
        content_hash=content_hash,
        byte_size=byte_size,
    )

    try:
        db.add(doc)
        await db.flush() # allocates doc.id

        if canonical_doc_id is None:
            canonical_doc_id = doc.id

        duplicate_status = "unique" if existing is None else "member"
        _set_ingestion_metadata(
            doc, ingestion_mode=mode,
            duplicate_group_id=group.id, duplicate_status=duplicate_status,
            canonical_doc_id=canonical_doc_id,
        )

        dst = await asyncio.to_thread(
            lambda: _copy_raw_into_doc(src_path, doc.id, original_filename)
        )
        doc.source_uri = to_output_uri(dst)
        await db.commit()

        outcome: IngestOutcome = "created" if existing is None else "duplicate_created"
        logger.info(
            f"[Ingestion] Created document doc_id={doc.id} group_id={group.id} "
            f"canonical_doc_id={canonical_doc_id} outcome={outcome} mode={mode}"
        )
        return IngestResult(
            doc_id=doc.id, outcome=outcome,
            hash_algo=hash_algo, content_hash=content_hash,
            duplicate_group_id=group.id, canonical_doc_id=canonical_doc_id,
        )

    except Exception:
        try:
            await db.rollback()
        except Exception as rb_err:
            logger.error(f"[Ingestion] Rollback failed: {rb_err}")

        if dst is not None:
            async def _unlink() -> None:
                try:
                    if dst.exists():
                        dst.unlink()
                except Exception:
                    pass
            await asyncio.to_thread(_unlink)

        raise



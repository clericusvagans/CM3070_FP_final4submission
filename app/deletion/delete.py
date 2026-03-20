# app/deletion/delete.py

import asyncio
import shutil
from pathlib import Path
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import Document, DocStatus
from app.utils.logger import logger
from app.utils.storage import doc_dir


async def _remove_tree_offloop(path: Path) -> None:
    """Recursive directory removal (best-effort), run off the event loop."""
    def _remove():
        try:
            if path.exists():
                shutil.rmtree(path)
        except Exception as e:
            logger.warning(f"[Delete] Filesystem cleanup failed path={path} err={e}")
    await asyncio.to_thread(_remove)


async def delete_document_async(
    db: AsyncSession,
    doc_id: int,
    *,
    idempotent: bool = True,
) -> bool:
    """
    Hard-delete a document and all owned data + filesystem artifacts.

    Deletion order:
        1. Mark document as `deleting` (commit) to gate concurrent workers.
           Workers must check for this status and abort to prevent new
           processing from starting on a document being deleted.
        2. Delete the document row (DB CASCADE removes all children).
        3. Commit.
        4. Best-effort delete the filesystem directory.

    Args:
        db:         AsyncSession.
        doc_id:     Document primary key.
        idempotent: If True, a non-existent document is treated as success.
                    If False, raises `KeyError`.

    Returns:
        True if the document existed and was deleted; False if it did not exist.
    """
    res = await db.execute(select(Document).where(Document.id == doc_id))
    doc = res.scalar_one_or_none()

    if doc is None:
        logger.info(f"[Delete] doc_id={doc_id} not found")
        if idempotent:
            await _remove_tree_offloop(doc_dir(doc_id))
            return False
        raise KeyError(f"Document {doc_id} not found")

    try:
        doc.status = DocStatus.deleting
        await db.commit()
    except Exception:
        await db.rollback()
        # Status update failure is non-fatal; proceed with hard delete.

    await db.delete(doc)
    await db.commit()
    logger.info(f"[Delete] DB delete committed doc_id={doc_id}")

    await _remove_tree_offloop(doc_dir(doc_id))
    logger.info(f"[Delete] Filesystem cleanup attempted doc_id={doc_id}")

    return True


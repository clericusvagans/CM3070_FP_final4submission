# app/routes/documents.py

"""
Document metadata and inspection endpoints.
Uploading and ingestion are handled by /ingest/... endpoints.
"""

from fastapi import APIRouter, Depends, HTTPException, Response
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from sqlalchemy.orm import selectinload

from app.db import get_session
from app.models import Document, Page
from app.utils.logger import logger
from app.deletion.delete import delete_document_async


router = APIRouter(prefix="/documents", tags=["documents"])


@router.get("/")
async def list_documents(db: AsyncSession = Depends(get_session)):
    """List all documents with minimal metadata (id, filename, status, page count)."""
    stmt = select(Document).options(selectinload(Document.pages))
    result = await db.execute(stmt)
    docs = result.scalars().all()
    return [
        {"id": d.id, "filename": d.filename, "status": d.status, "pages": len(d.pages)}
        for d in docs
    ]


@router.get("/{doc_id}")
async def get_document(doc_id: int, db: AsyncSession = Depends(get_session)):
    """Return detailed metadata for a single document."""
    stmt = (
        select(Document)
        .options(selectinload(Document.pages))
        .where(Document.id == doc_id)
    )
    result = await db.execute(stmt)
    doc = result.scalar_one_or_none()

    if doc is None:
        raise HTTPException(404, "Document not found")

    return {
        "id": doc.id,
        "filename": doc.filename,
        "status": doc.status,
        "created_at": doc.created_at,
        "updated_at": doc.updated_at,
        "pages": [
            {"id": p.id, "page_number": p.page_number, "md_uri": p.md_uri}
            for p in sorted(doc.pages, key=lambda x: x.page_number)
        ],
    }


@router.get("/{doc_id}/pages/{page_number}")
async def get_page(doc_id: int, page_number: int, db: AsyncSession = Depends(get_session)):
    """Return metadata for a specific page of a document."""
    stmt = select(Page).where(Page.document_id == doc_id, Page.page_number == page_number)
    result = await db.execute(stmt)
    page = result.scalar_one_or_none()

    if page is None:
        doc = await db.get(Document, doc_id)
        if doc is None:
            raise HTTPException(404, "Document not found")
        raise HTTPException(404, "Page not found")

    return {
        "id": page.id,
        "page_number": page.page_number,
        "width": page.width,
        "height": page.height,
        "md_uri": page.md_uri,
    }


@router.delete("/{doc_id}", status_code=204)
async def delete_document(
    doc_id: int,
    response: Response,
    db: AsyncSession = Depends(get_session),
):
    """
    Hard-delete a document and all owned artifacts.

    Idempotent: returns 204 whether or not the document existed.
    The `X-Deleted-Existed` response header indicates whether a DB row was present.
    """
    existed = await delete_document_async(db, doc_id, idempotent=True)
    response.headers["X-Deleted-Existed"] = "1" if existed else "0"
    return None


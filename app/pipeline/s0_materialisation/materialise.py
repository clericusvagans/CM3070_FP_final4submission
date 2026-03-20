# app/pipeline/s0_materialisation/materialise.py

"""
Stage S0: Document materialisation.

Ensures every Page row for a Document exists and has a persisted raster page
image with its URI recorded in page_metadata["image_uri"]. This is the
mandatory first stage; all subsequent stages assume it has completed successfully.

Idempotency
-----------
If Page rows already exist and each has a non-empty image_uri, the stage
returns immediately without re-materialising.
If Page rows exist but some lack image_uri, images are regenerated and metadata
is backfilled.

Artifacts written
-----------------
    <OUTPUT_ROOT>/documents/<doc_id:09d>/<page_number:04d>/p<page_number>_source.png
"""

from typing import Any

from sqlalchemy.orm import Session

from app.models import Document, Page
from app.utils.logger import logger
from app.utils.storage import resolve_local_path, split_pdf_or_image, write_page_images


def _infer_input_kind(source_uri: str) -> str:
    """
    Infer coarse input kind from file extension.
    Returns `"pdf"`, `"image"`, or `"unknown"` (treated as image-like).
    """
    ext = resolve_local_path(source_uri).suffix.lower()
    if ext == ".pdf":
        return "pdf"
    if ext in {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}:
        return "image"
    return "unknown"


def _has_image_uri(page: Page) -> bool:
    """Return True if page_metadata contains a non-empty image_uri string."""
    meta: dict[str, Any] = page.page_metadata or {}
    v = meta.get("image_uri")
    return isinstance(v, str) and len(v) > 0


def materialise_document_pages(db: Session, doc: Document) -> list[Page]:
    """
    Stage S0: ensure Page rows and persistent page images exist for the document.

    DPI handling
    ------------
    doc.dpi    Requested / representative DPI at the document level (default 300).
               Set from the median of per-page DPIs if not already present.
    page.dpi   Actual DPI used or inferred per page.
    page_metadata["dpi"]  Same value, stored for traceability.

    JSONB copy-on-write
    -------------------
    page_metadata is updated by building a new dict and reassigning the
    attribute. In-place mutation of the existing dict (e.g. setdefault() on
    the dict object directly) is silently ignored by SQLAlchemy's change
    tracking for JSONB columns — reassignment is required.

    Returns:
        Page ORM objects ordered by page_number ascending.
    """
    if not doc.source_uri:
        raise ValueError(f"Document {doc.id} has no source_uri; cannot materialise pages")

    meta = dict(doc.doc_metadata or {})
    meta.setdefault("input_kind", _infer_input_kind(doc.source_uri))
    doc.doc_metadata = meta

    pages: list[Page] = (
        db.query(Page)
        .filter(Page.document_id == doc.id)
        .order_by(Page.page_number)
        .all()
    )

    if pages and all(_has_image_uri(p) for p in pages):
        logger.info(f"[Orchestrator][S0] Already materialised doc_id={doc.id} pages={len(pages)}")
        return pages

    requested_dpi = int(doc.dpi or 300)
    logger.info(
        f"[Orchestrator][S0] Materializing doc_id={doc.id} "
        f"source_uri={doc.source_uri} requested_dpi={requested_dpi}"
    )

    page_imgs_bgr, page_dpis = split_pdf_or_image(doc.source_uri, dpi=requested_dpi)

    if not page_imgs_bgr:
        raise RuntimeError(f"[Orchestrator][S0] No pages produced for doc_id={doc.id}")
    if len(page_imgs_bgr) != len(page_dpis):
        raise RuntimeError(
            f"[Orchestrator][S0] Internal error: "
            f"page_imgs({len(page_imgs_bgr)}) != page_dpis({len(page_dpis)}) "
            f"doc_id={doc.id}"
        )

    image_uris = write_page_images(doc.id, page_imgs_bgr)

    by_page_num: dict[int, Page] = {int(p.page_number): p for p in pages}

    for idx0, (img, img_uri, dpi_used) in enumerate(zip(page_imgs_bgr, image_uris, page_dpis)):
        page_num = idx0 + 1
        h, w    = img.shape[:2]

        if page_num in by_page_num:
            p = by_page_num[page_num]
            if not p.width:
                p.width = w
            if not p.height:
                p.height = h
            if getattr(p, "dpi", None) in (None, 0):
                p.dpi = int(dpi_used)
            pm = dict(p.page_metadata or {})
            pm.setdefault("image_uri", img_uri)
            pm.setdefault("dpi", int(dpi_used))
            p.page_metadata = pm
        else:
            p = Page(
                document_id=doc.id,
                page_number=page_num,
                width=w,
                height=h,
                dpi=int(dpi_used),
                page_metadata={"image_uri": img_uri, "dpi": int(dpi_used)},
            )
            db.add(p)
            by_page_num[page_num] = p

    if not doc.dpi and page_dpis:
        s       = sorted(int(x) for x in page_dpis)
        doc.dpi = int(s[len(s) // 2])
        logger.info(
            f"[Orchestrator][S0] Set doc.dpi from per-page DPIs "
            f"doc_id={doc.id} doc.dpi={doc.dpi}"
        )

    db.flush()

    pages = (
        db.query(Page)
        .filter(Page.document_id == doc.id)
        .order_by(Page.page_number)
        .all()
    )

    logger.info(f"[Orchestrator][S0] Materialization complete doc_id={doc.id} pages={len(pages)}")
    return pages



# app/routes/ui.py

from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.templating import Jinja2Templates

from sqlalchemy import func, literal, select
from sqlalchemy.ext.asyncio import AsyncSession

from pathlib import Path

from app.db import get_session
from app.models import BlockMT, Document, Page, PageBlock, PageMT

from app.utils.storage import resolve_local_path


router    = APIRouter(prefix="/ui", tags=["ui"])
templates = Jinja2Templates(directory="app/templates")

# ts_headline options for the browser UI.
# StartSel/StopSel inject <mark> tags so the template can highlight the match.
# MaxWords/MinWords kept in sync with the JSON API (_HEADLINE_OPTS in search.py).
_HEADLINE_OPTS = (
    "StartSel=<mark>, StopSel=</mark>, "
    "MaxWords=40, MinWords=15, ShortWord=2, "
    "MaxFragments=2, FragmentDelimiter= … "
)


@router.get("/search", response_class=HTMLResponse)
async def search_ui(
    request: Request,
    q: str = Query("", description="Search query"),
    target: str = Query("all", pattern="^(all|original|page_mt|block_mt|block_original)$"),
    limit: int = Query(50, ge=1, le=100),
    db: AsyncSession = Depends(get_session),
):
    """
    Server-rendered search UI.

    Searches page markdown (original), page translations (page_mt),
    block translations (block_mt), and block original text (block_original).
    Uses ts_headline() for hit-centred snippets with <mark> highlighting.
    """
    results: list[dict] = []
    q = q.strip()

    if q:
        ts_query = func.plainto_tsquery("simple", q)

        if target in ("all", "original"):
            stmt = (
                select(
                    literal("original").label("source"),
                    Document.id.label("document_id"),
                    Page.id.label("page_id"),
                    Page.page_number.label("page_number"),
                    literal(None).label("lang"),
                    func.ts_headline(
                        "simple",
                        func.coalesce(Page.markdown, ""),
                        ts_query,
                        _HEADLINE_OPTS,
                    ).label("snippet"),
                )
                .join(Page, Page.document_id == Document.id)
                .where(func.to_tsvector("simple", func.coalesce(Page.markdown, "")).op("@@")(ts_query))
                .order_by(Document.id.desc(), Page.page_number.asc())
                .limit(limit)
            )
            for r in (await db.execute(stmt)).mappings().all():
                results.append({
                    "source":               r["source"],
                    "document_id":          r["document_id"],
                    "page_id":              r["page_id"],
                    "page_number":          r["page_number"],
                    "lang":                 r["lang"],
                    "snippet":              r["snippet"],
                    "original_page_url":    f"/ui/pages/{r['page_id']}",
                    "translated_page_url":  f"/ui/pages/{r['page_id']}/translation/en",
                })

        if target in ("all", "page_mt"):
            stmt = (
                select(
                    literal("page_mt").label("source"),
                    Document.id.label("document_id"),
                    Page.id.label("page_id"),
                    Page.page_number.label("page_number"),
                    PageMT.lang.label("lang"),
                    func.ts_headline(
                        "simple",
                        func.coalesce(PageMT.text, ""),
                        ts_query,
                        _HEADLINE_OPTS,
                    ).label("snippet"),
                )
                .join(Page, PageMT.page_id == Page.id)
                .join(Document, Page.document_id == Document.id)
                .where(func.to_tsvector("simple", func.coalesce(PageMT.text, "")).op("@@")(ts_query))
                .order_by(Document.id.desc(), Page.page_number.asc())
                .limit(limit)
            )
            for r in (await db.execute(stmt)).mappings().all():
                results.append({
                    "source":               r["source"],
                    "document_id":          r["document_id"],
                    "page_id":              r["page_id"],
                    "page_number":          r["page_number"],
                    "lang":                 r["lang"],
                    "snippet":              r["snippet"],
                    "original_page_url":    f"/ui/pages/{r['page_id']}",
                    "translated_page_url":  f"/ui/pages/{r['page_id']}/translation/{r['lang']}",
                })

        if target in ("all", "block_mt"):
            stmt = (
                select(
                    literal("block_mt").label("source"),
                    Document.id.label("document_id"),
                    Page.id.label("page_id"),
                    Page.page_number.label("page_number"),
                    BlockMT.lang.label("lang"),
                    PageBlock.block_index.label("block_index"),
                    func.ts_headline(
                        "simple",
                        func.coalesce(BlockMT.text, ""),
                        ts_query,
                        _HEADLINE_OPTS,
                    ).label("snippet"),
                )
                .join(PageBlock, BlockMT.block_id == PageBlock.id)
                .join(Page, PageBlock.page_id == Page.id)
                .join(Document, Page.document_id == Document.id)
                .where(func.to_tsvector("simple", func.coalesce(BlockMT.text, "")).op("@@")(ts_query))
                .order_by(Document.id.desc(), Page.page_number.asc(), BlockMT.id.asc())
                .limit(limit)
            )
            for r in (await db.execute(stmt)).mappings().all():
                results.append({
                    "source":               r["source"],
                    "document_id":          r["document_id"],
                    "page_id":              r["page_id"],
                    "page_number":          r["page_number"],
                    "lang":                 r["lang"],
                    "block_index":          r["block_index"],
                    "snippet":              r["snippet"],
                    "original_page_url":    f"/ui/pages/{r['page_id']}",
                    "translated_page_url":  f"/ui/pages/{r['page_id']}/translation/{r['lang']}",
                })

        if target in ("all", "block_original"):
            stmt = (
                select(
                    literal("block_original").label("source"),
                    Document.id.label("document_id"),
                    Page.id.label("page_id"),
                    Page.page_number.label("page_number"),
                    literal(None).label("lang"),
                    PageBlock.block_index.label("block_index"),
                    func.ts_headline(
                        "simple",
                        func.coalesce(PageBlock.text, ""),
                        ts_query,
                        _HEADLINE_OPTS,
                    ).label("snippet"),
                )
                .join(Page, PageBlock.page_id == Page.id)
                .join(Document, Page.document_id == Document.id)
                .where(func.to_tsvector("simple", func.coalesce(PageBlock.text, "")).op("@@")(ts_query))
                .order_by(Document.id.desc(), Page.page_number.asc(), PageBlock.id.asc())
                .limit(limit)
            )
            for r in (await db.execute(stmt)).mappings().all():
                results.append({
                    "source":               r["source"],
                    "document_id":          r["document_id"],
                    "page_id":              r["page_id"],
                    "page_number":          r["page_number"],
                    "lang":                 r["lang"],
                    "block_index":          r["block_index"],
                    "snippet":              r["snippet"],
                    "original_page_url":    f"/ui/pages/{r['page_id']}",
                    "translated_page_url":  f"/ui/pages/{r['page_id']}/translation/en",
                })

    return templates.TemplateResponse(
        "search.html",
        {
            "request": request,
            "q":       q,
            "target":  target,
            "limit":   limit,
            "results": results,
        },
    )


@router.get("/pages/{page_id}", response_class=HTMLResponse)
async def page_view(
    request: Request,
    page_id: int,
    db: AsyncSession = Depends(get_session),
):
    page = await db.get(Page, page_id)
    if not page:
        return HTMLResponse("Page not found", status_code=404)

    image_uri = (page.page_metadata or {}).get("image_uri")

    return templates.TemplateResponse(
        "page.html",
        {
            "request":     request,
            "page_id":     page.id,
            "page_number": page.page_number,
            "document_id": page.document_id,
            "image_url":   f"/ui/pages/{page.id}/image" if image_uri else None,
        },
    )


@router.get("/pages/{page_id}/image")
async def page_image(
    page_id: int,
    db: AsyncSession = Depends(get_session),
):
    page = await db.get(Page, page_id)
    if not page:
        return HTMLResponse("Page not found", status_code=404)

    image_uri = (page.page_metadata or {}).get("image_uri")
    if not image_uri:
        return HTMLResponse("No image available", status_code=404)

    local_path = resolve_local_path(image_uri)
    if not local_path or not Path(local_path).exists():
        return HTMLResponse("Image not found on disk", status_code=404)

    return FileResponse(local_path, media_type="image/png")


@router.get("/pages/{page_id}/translation/{lang}", response_class=HTMLResponse)
async def page_translation_view(
    request: Request,
    page_id: int,
    lang: str,
    db: AsyncSession = Depends(get_session),
):
    """
    Show the translated page text for a given page/language.
    Route is intentionally stable: every result type can link here.
    """
    page = await db.get(Page, page_id)
    if not page:
        return HTMLResponse("Page not found", status_code=404)

    stmt    = select(PageMT).where(PageMT.page_id == page_id, PageMT.lang == lang).limit(1)
    page_mt = (await db.execute(stmt)).scalar_one_or_none()

    return templates.TemplateResponse(
        "page_translation.html",
        {
            "request":           request,
            "page_id":           page.id,
            "page_number":       page.page_number,
            "document_id":       page.document_id,
            "lang":              lang,
            "text":              page_mt.text if page_mt else None,
            "uri":               page_mt.uri  if page_mt else None,
            "original_page_url": f"/ui/pages/{page.id}",
            "has_translation":   page_mt is not None,
        },
    )




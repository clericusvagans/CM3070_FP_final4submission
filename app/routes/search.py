# app/routes/search.py

"""
Keyword / FTS search endpoints across original page text and machine translations.

All four search paths use:
    - SQLAlchemy ORM query API
    - Postgres to_tsvector() / plainto_tsquery() via func.*
    - ts_headline() for hit-centred snippets
    - GIN indexes on all searched text columns

GIN indexes in use:
    ix_pages_markdown_fts        to_tsvector('simple', pages.markdown)
    ix_page_mt_text_fts          to_tsvector('simple', page_mt.text)
    ix_block_mt_text_fts         to_tsvector('simple', block_mt.text)
    ix_page_blocks_text_fts      to_tsvector('simple', page_blocks.text)
"""

from fastapi import APIRouter, Depends, Query
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import get_session
from app.models import BlockMT, Document, Page, PageBlock, PageMT

router = APIRouter()

# ts_headline options for the JSON API.
# No StartSel/StopSel — callers apply their own markup.
# MaxWords/MinWords kept in sync with _HEADLINE_OPTS in ui.py.
_HEADLINE_OPTS = (
    "MaxWords=40, MinWords=15, ShortWord=2, "
    "MaxFragments=2, FragmentDelimiter= … , HighlightAll=FALSE"
)
_HEADLINE_OPTS_BLOCK = (
    "MaxWords=60, MinWords=20, ShortWord=2, "
    "MaxFragments=2, FragmentDelimiter= … , HighlightAll=FALSE"
)


@router.get("/search/keyword")
async def keyword_search(
    q: str = Query(..., min_length=1, description="Keyword or phrase to search for"),
    scope: str = Query("both", pattern="^(original|mt|both)$"),
    level: str = Query("block", pattern="^(block|page)$"),
    limit: int = Query(50, ge=1, le=50),
    db: AsyncSession = Depends(get_session),
) -> dict:
    """
    Full-text search across originals and/or machine translations.

    scope=original  Searches page markdown (page level) or block original text
                    (block level). Both backed by GIN FTS indexes.
    scope=mt        Searches PageMT (page level) or BlockMT (block level).
    scope=both      Searches all applicable sources and de-duplicates results.

    level=block     Returns block-level hits (more precise context).
    level=page      Returns page-level hits.

    Result limit is hard-capped at 50.
    """
    q_norm   = q.strip()
    ts_query = func.plainto_tsquery("simple", q_norm)
    hits: list[dict] = []

    if level == "block":
        if scope in {"mt", "both"}:
            hits.extend(await _search_block_mt(db, ts_query, limit))
        if scope in {"original", "both"}:
            hits.extend(await _search_block_original(db, ts_query, limit))
    else:  # level == "page"
        if scope in {"mt", "both"}:
            hits.extend(await _search_page_mt(db, ts_query, limit))
        if scope in {"original", "both"}:
            hits.extend(await _search_page_original(db, ts_query, limit))

    seen: set = set()
    out:  list[dict] = []
    for h in hits:
        key = (h.get("source"), h.get("document_id"), h.get("page_id"), h.get("block_id"), h.get("lang"))
        if key in seen:
            continue
        seen.add(key)
        out.append(h)
        if len(out) >= limit:
            break

    return {"query": q_norm, "scope": scope, "level": level, "limit": limit, "hits": out}


async def _search_page_original(db: AsyncSession, ts_query, limit: int) -> list[dict]:
    stmt = (
        select(
            Document.id.label("document_id"),
            Page.id.label("page_id"),
            Page.page_number.label("page_number"),
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
    rows = (await db.execute(stmt)).mappings().all()
    return [
        {
            "source":      "page_original",
            "document_id": int(r["document_id"]),
            "page_id":     int(r["page_id"]),
            "page_number": int(r["page_number"]),
            "block_id":    None,
            "lang":        None,
            "context":     r["snippet"] or "",
        }
        for r in rows
    ]


async def _search_page_mt(db: AsyncSession, ts_query, limit: int) -> list[dict]:
    stmt = (
        select(
            Document.id.label("document_id"),
            Page.id.label("page_id"),
            Page.page_number.label("page_number"),
            PageMT.lang.label("lang"),
            PageMT.model.label("model"),
            PageMT.translated_at.label("translated_at"),
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
    rows = (await db.execute(stmt)).mappings().all()
    return [
        {
            "source":        "page_mt",
            "document_id":   int(r["document_id"]),
            "page_id":       int(r["page_id"]),
            "page_number":   int(r["page_number"]),
            "block_id":      None,
            "lang":          str(r["lang"]),
            "model":         str(r["model"]) if r.get("model") else None,
            "translated_at": r["translated_at"].isoformat() if r.get("translated_at") else None,
            "context":       r["snippet"] or "",
        }
        for r in rows
    ]


async def _search_block_mt(db: AsyncSession, ts_query, limit: int) -> list[dict]:
    stmt = (
        select(
            Document.id.label("document_id"),
            Page.id.label("page_id"),
            Page.page_number.label("page_number"),
            BlockMT.id.label("block_mt_id"),
            PageBlock.id.label("block_id"),
            PageBlock.block_index.label("block_index"),
            BlockMT.lang.label("lang"),
            BlockMT.model.label("model"),
            BlockMT.translated_at.label("translated_at"),
            func.ts_headline(
                "simple",
                func.coalesce(BlockMT.text, ""),
                ts_query,
                _HEADLINE_OPTS_BLOCK,
            ).label("snippet"),
        )
        .join(PageBlock, BlockMT.block_id == PageBlock.id)
        .join(Page, PageBlock.page_id == Page.id)
        .join(Document, Page.document_id == Document.id)
        .where(func.to_tsvector("simple", func.coalesce(BlockMT.text, "")).op("@@")(ts_query))
        .order_by(Document.id.desc(), Page.page_number.asc(), BlockMT.id.asc())
        .limit(limit)
    )
    rows = (await db.execute(stmt)).mappings().all()
    return [
        {
            "source":        "block_mt",
            "document_id":   int(r["document_id"]),
            "page_id":       int(r["page_id"]),
            "page_number":   int(r["page_number"]),
            "block_id":      int(r["block_id"]),
            "block_index":   int(r["block_index"]),
            "lang":          str(r["lang"]),
            "model":         str(r["model"]) if r.get("model") else None,
            "translated_at": r["translated_at"].isoformat() if r.get("translated_at") else None,
            "context":       r["snippet"] or "",
        }
        for r in rows
    ]


async def _search_block_original(db: AsyncSession, ts_query, limit: int) -> list[dict]:
    stmt = (
        select(
            Document.id.label("document_id"),
            Page.id.label("page_id"),
            Page.page_number.label("page_number"),
            PageBlock.id.label("block_id"),
            PageBlock.block_index.label("block_index"),
            func.ts_headline(
                "simple",
                func.coalesce(PageBlock.text, ""),
                ts_query,
                _HEADLINE_OPTS_BLOCK,
            ).label("snippet"),
        )
        .join(Page, PageBlock.page_id == Page.id)
        .join(Document, Page.document_id == Document.id)
        .where(func.to_tsvector("simple", func.coalesce(PageBlock.text, "")).op("@@")(ts_query))
        .order_by(Document.id.desc(), Page.page_number.asc(), PageBlock.id.asc())
        .limit(limit)
    )
    rows = (await db.execute(stmt)).mappings().all()
    return [
        {
            "source":      "block_original",
            "document_id": int(r["document_id"]),
            "page_id":     int(r["page_id"]),
            "page_number": int(r["page_number"]),
            "block_id":    int(r["block_id"]),
            "block_index": int(r["block_index"]),
            "lang":        None,
            "context":     r["snippet"] or "",
        }
        for r in rows
    ]




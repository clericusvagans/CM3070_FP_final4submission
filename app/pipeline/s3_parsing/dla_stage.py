# app/pipeline/s3_parsing/dla_stage.py

"""
Stage S3: Parsing (DLA/OCR) for the standard DB-backed pipeline.

Responsibilities
----------------
For each page:
    1. Resolve the rendered page image URI from page.page_metadata.
    2. Run PP-StructureV3 parsing (layout analysis + OCR/structure).
    3. Write stable page-level artifacts to the document output directory.
    4. Persist normalized blocks into PageBlock rows.
    5. Update page.page_metadata["paddle"] with an auditable parse summary.
    6. Set Page.md_uri, Page.markdown (plain TEXT, GIN-indexed for FTS), and
       Page.text_layer (canonical MT input; ordered PageBlock text).

Language handling
-----------------
The PaddleParser singleton is language-neutral (cfg.lang=None) so the worker
is not pinned to the first language it encounters. Language is passed per page,
resolved in this order:
    1. Page.primary_lang (refreshed from DB; authoritative per-page detection)
    2. Document.primary_lang
    3. First entry of Document.declared_langs
    4. "en" (final default)

The singleton caches PP-StructureV3 pipelines keyed by language within the
worker process, so repeated languages reuse the same in-memory model.

IMPORTANT: Since subsequent stages require GPU headroom (e.g. CT2 MT), the
Paddle singleton must be released after this stage so model weights do not
remain resident in VRAM for the lifetime of the worker process.

Design notes
------------
- Idempotent per (doc_id, page_id): re-running replaces PageBlock rows.
- Blocks for a page are deleted immediately before re-parsing that page.
  This means an interrupted run leaves already-completed pages intact.
- One db.flush() at the end of the stage (all ORM work batched).
- strict=False (default): page failures are recorded in page metadata and
  the stage continues. strict=True aborts on the first failure.
"""

from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session

from app.models import Document, Page, PageBlock, PageImage
from app.pipeline.s3_parsing.paddle_parse import PaddleParseConfig
from app.pipeline.s3_parsing.paddle_singleton import get_paddle_parser, release_paddle_parser
from app.pipeline.s3_parsing.artifacts import (
    persist_page_artifacts_simple,
    write_empty_page_artifacts_simple,
)
from app.utils import storage
from app.utils.logger import logger


def parse_and_persist_document(
    db: Session,
    *,
    doc: Document,
    pages: list[Page],
    device: str = "gpu",
    strict: bool = False,
) -> dict[str, Any]:
    """
    Parse all rendered page images for a document and persist the results.

    Args:
        db:      Sync SQLAlchemy session (worker context).
        doc:     ORM Document.
        pages:   ORM Pages for the document (materialized by S0).
        device:  Paddle device selector ("gpu" | "cpu").
        strict:  If True, any page failure raises and stops the stage.

    Returns:
        Summary dict with page/block counts and parser configuration info.
    """
    if not pages:
        logger.warning(f"[DLA] doc_id={doc.id} has no pages; skipping parse stage")
        return {"doc_id": doc.id, "pages": 0, "parsed_pages": 0, "failed_pages": 0}

    declared_langs_for_log = doc.declared_langs or []

    # Keep singleton config language-neutral; language is resolved per page.
    cfg = PaddleParseConfig(device=device, lang=None)
    parser = get_paddle_parser(cfg=cfg)

    parsed_pages = 0
    failed_pages = 0
    total_blocks = 0

    for page in pages:
        page_number = int(page.page_number)

        # Delete existing blocks immediately before re-parsing this page so that
        # an interrupted run leaves already-completed pages intact.
        db.query(PageBlock).filter(PageBlock.page_id == page.id).delete(
            synchronize_session=False
        )

        # ─── Resolve page image URI ───
        pm = dict(page.page_metadata or {})
        image_uri = pm.get("image_uri") or pm.get("image_path") or pm.get("img_uri")

        if not isinstance(image_uri, str) or not image_uri:
            failed_pages += 1
            pm["paddle"] = {
                "parse_ok": False,
                "error": "Missing page image URI in page_metadata (expected key 'image_uri')",
                "pipeline": cfg.pipeline,
                "device": cfg.device,
                "lang": None,
                "declared_langs": declared_langs_for_log,
            }
            page.page_metadata = pm
            page.md_uri = None
            page.markdown = None
            page.text_layer = None
            if strict:
                raise RuntimeError(
                    f"[DLA] Missing image URI for doc_id={doc.id} page_id={page.id}"
                )
            continue

        try:
            img_path = storage.resolve_local_path(image_uri)
        except Exception as e:
            failed_pages += 1
            pm["paddle"] = {
                "parse_ok":  False,
                "error": f"Failed to resolve image URI: {e}",
                "image_uri": image_uri,
                "pipeline": cfg.pipeline,
                "device": cfg.device,
                "lang": None,
                "declared_langs": declared_langs_for_log,
            }
            page.page_metadata = pm
            page.md_uri = None
            page.markdown = None
            page.text_layer = None
            if strict:
                raise
            continue

        # ─── Per-page language (refreshed from DB each page) ───
        db.refresh(page, attribute_names=["primary_lang"])
        page_lang = (
            page.primary_lang
            or doc.primary_lang
            or ((doc.declared_langs or [])[0] if doc.declared_langs else "en")
        )

        # ─── Parse + persist artifacts ───
        try:
            parsed = parser.parse(str(img_path), lang=page_lang)

            meta = persist_page_artifacts_simple(
                doc_id=doc.id,
                page_number=page_number,
                pipeline_name=cfg.pipeline,
                parsed=parsed,
            )

            blocks = parsed.get("blocks") or []

            blocks_written = 0
            for block_index, b in enumerate(blocks):
                bbox = b.get("bbox")
                if not (isinstance(bbox, list) and len(bbox) == 4):
                    continue
                db.add(PageBlock(
                    page_id=page.id,
                    block_index=block_index,
                    reading_order=b.get("reading_order"),
                    category=str(b.get("category") or "Unknown"),
                    bbox_px=bbox,
                    text=str(b.get("text") or ""),
                    engine=cfg.pipeline,
                ))
                blocks_written += 1

            parsed_pages += 1
            total_blocks += blocks_written

            md_uri = storage.to_output_uri(meta["page_md_path"])
            marked_md_uri = storage.to_output_uri(meta["page_marked_md_path"])
            md_text: str = meta["page_md_path"].read_text(encoding="utf-8").strip()

            pm["paddle"] = {
                "parse_ok":                True,
                "pipeline":                cfg.pipeline,
                "device":                  cfg.device,
                "lang":                    page_lang,
                "num_blocks":              int(meta["num_blocks"]),
                "num_blocks_persisted":    blocks_written,
                "markdown_chars":          int(meta["markdown_chars"]),
                "marked_markdown_chars":   int(meta.get("marked_markdown_chars", 0)),
                "page_json_uri":           storage.to_output_uri(meta["page_json_path"]),
                "res_json_uri":            storage.to_output_uri(meta["res_json_path"]),
                "page_md_uri":             md_uri,
                "page_marked_md_uri":      marked_md_uri,
                "image_uri":               image_uri,
            }
            page.page_metadata = pm
            page.md_uri = md_uri
            page.markdown = md_text or None  # store NULL rather than empty string

            # text_layer: concatenate block text in reading order (MT input).
            ordered = sorted(
                [x for x in blocks if isinstance(x, dict)],
                key=lambda d: (
                    int(d["reading_order"]) if d.get("reading_order") is not None else 10**9,
                    int(d["block_index"]) if d.get("block_index") is not None else 10**9,
                ),
            )
            page.text_layer = (
                "\n".join(t for t in (str(d.get("text") or "").strip() for d in ordered) if t)
                or None
            )

            # ─── Persist markdown images as PageImage rows ───
            # PageImage.filename stores the markdown-relative path key (e.g. "imgs/table_0.png")
            # so that markdown references ![](imgs/table_0.png) can be resolved back to the row.
            db.query(PageImage).filter(PageImage.page_id == page.id).delete(
                synchronize_session=False
            )
            markdown_images = (parsed.get("markdown") or {}).get("markdown_images") or {}
            page_root       = storage.page_dir(doc.id, page_number)

            for rel_path_str in markdown_images:
                try:
                    abs_path = page_root / str(rel_path_str)
                    db.add(PageImage(
                        page_id=page.id,
                        filename=str(rel_path_str),
                        uri=storage.to_output_uri(abs_path),
                        data=abs_path.read_bytes() if abs_path.exists() else None,
                    ))
                except Exception:
                    continue

        except Exception as e:
            failed_pages += 1
            empty = write_empty_page_artifacts_simple(
                doc_id=doc.id, page_number=page_number, pipeline_name=cfg.pipeline
            )
            pm["paddle"] = {
                "parse_ok":     False,
                "error":        str(e),
                "pipeline":     cfg.pipeline,
                "device":       cfg.device,
                "lang":         page_lang,
                "page_json_uri": storage.to_output_uri(empty["page_json_path"]),
                "res_json_uri":  storage.to_output_uri(empty["res_json_path"]),
                "page_md_uri":   storage.to_output_uri(empty["page_md_path"]),
                "image_uri":     image_uri,
            }
            page.page_metadata = pm
            page.md_uri = None
            page.markdown = None
            page.text_layer = None

            logger.warning(
                f"[DLA] Parse failed doc_id={doc.id} page_number={page_number}: {e}"
            )
            if strict:
                raise

    # One flush for all ORM work.
    db.flush()

    logger.info(
        f"[DLA] doc_id={doc.id} declared_langs={declared_langs_for_log} "
        f"pages={len(pages)} parsed={parsed_pages} failed={failed_pages} blocks={total_blocks}"
    )

    # Release the Paddle singleton so GPU VRAM is freed before subsequent stages
    # (e.g. CT2 MT) that also need GPU headroom.
    if cfg.device == "gpu":
        try:
            release_paddle_parser()
        except Exception as e:
            logger.warning(f"[DLA] Failed to release Paddle singleton doc_id={doc.id}: {e}")

    return {
        "doc_id":           doc.id,
        "pages":            len(pages),
        "parsed_pages":     parsed_pages,
        "failed_pages":     failed_pages,
        "blocks_persisted": total_blocks,
        "parser":           {"pipeline": cfg.pipeline, "device": cfg.device, "lang": None},
        "declared_langs":   declared_langs_for_log,
    }





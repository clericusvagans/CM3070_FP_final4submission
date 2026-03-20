# app/pipeline/s3_parsing/artifacts.py

import json
from pathlib import Path
from typing import Any, Mapping

from app.utils import storage


def persist_page_artifacts_simple(
    *,
    doc_id: int,
    page_number: int,
    pipeline_name: str,
    parsed: Mapping[str, Any],
    page_width: int = 2481,
) -> dict[str, Any]:
    """
    Persist per-page artifacts produced by PaddleParser.parse().

    parsed contract
    ---------------
    parsed["blocks"]    List of normalized block dicts.
    parsed["markdown"]  res.markdown dict from Paddle:
                            markdown_texts (str), markdown_images (dict[str, PIL.Image]),
                            page_continuation_flags (tuple).
    parsed["img"]       res.img dict: layout_det_res (PIL.Image) + optional keys.
    parsed["raw_json"]  res.json converted to plain Python.

    Files written under the page folder
    ------------------------------------
    p<N>_page.json          Normalized blocks.
    p<N>_<pipeline>.json    Raw PP-StructureV3 output.
    p<N>_text.md            Original markdown from Paddle.
    p<N>_text_marked.md     Markdown with [#NNN]…[/#NNN] block markers for
                            unified single-pass translation + post-processing.
    imgs/*                  Markdown images (charts, tables rendered as images).
    LAYOUT/*                Layout/diagnostic visualizations.

    Returns a dict with counts and Path objects for all written artifacts.
    """
    from app.pipeline.s3_parsing.paddle_parse import get_marked_markdown_for_translation

    page_root = storage.page_dir(doc_id, page_number)
    layout_dir = storage.page_layout_dir(doc_id, page_number)
    prefix = storage.page_prefix(page_number)

    page_json_path = page_root / f"{prefix}_page.json"
    res_json_path = page_root / f"{prefix}_{pipeline_name}.json"
    page_md_path = page_root / f"{prefix}_text.md"
    page_marked_md_path = page_root / f"{prefix}_text_marked.md"

    blocks = list(parsed.get("blocks") or [])
    raw_json = parsed.get("raw_json") or {}
    markdown = parsed.get("markdown") or {}
    img_dict = parsed.get("img") or {}

    md_text = (markdown.get("markdown_texts") or "").strip()
    marked_md_text = get_marked_markdown_for_translation(parsed, page_width=page_width)

    # Markdown images (charts, tables rendered as images).
    # markdown_images is dict[relative_path_str, PIL.Image]; save each relative to page_root markdown references like ![](imgs/table_0.png) resolve correctly.
    markdown_images = markdown.get("markdown_images") or {}
    saved_markdown_images: list[Path] = []
    for rel_path_str, img_obj in markdown_images.items():
        img_save_path = page_root / rel_path_str
        img_save_path.parent.mkdir(parents=True, exist_ok=True)
        img_obj.save(str(img_save_path))
        saved_markdown_images.append(img_save_path)

    # Layout / diagnostic visualizations (best-effort).
    extra_img_paths: dict[str, Path] = {}
    for key, img_obj in img_dict.items():
        extra_path = layout_dir / f"{prefix}_img_{key}.png"
        try:
            img_obj.save(str(extra_path))
            extra_img_paths[key] = extra_path
        except Exception:
            pass

    page_json_path.write_text(json.dumps(blocks, ensure_ascii=False, indent=2), encoding="utf-8")
    res_json_path.write_text(json.dumps(raw_json, ensure_ascii=False, indent=2), encoding="utf-8")
    page_md_path.write_text(f"{md_text}\n" if md_text else "", encoding="utf-8")
    page_marked_md_path.write_text(marked_md_text, encoding="utf-8")

    return {
        "num_blocks": len(blocks),
        "markdown_chars": len(md_text),
        "marked_markdown_chars": len(marked_md_text),
        "page_json_path": page_json_path,
        "res_json_path": res_json_path,
        "page_md_path": page_md_path,
        "page_marked_md_path": page_marked_md_path,
        "markdown_image_paths": saved_markdown_images,
        "extra_img_paths": extra_img_paths,
    }


def write_empty_page_artifacts_simple(
    *,
    doc_id: int,
    page_number: int,
    pipeline_name: str,
) -> dict[str, Any]:
    """
    Write empty placeholder artifacts that satisfy the per-page file contract.

    Downstream consumers can rely on file existence and should inspect
    page_metadata["paddle"]["parse_ok"] to distinguish real results from
    placeholders. Returns the same keys as persist_page_artifacts_simple so
    callers can treat both code paths uniformly.
    """
    page_root = storage.page_dir(doc_id, page_number)
    prefix = storage.page_prefix(page_number)

    page_json_path = page_root / f"{prefix}_page.json"
    res_json_path = page_root / f"{prefix}_{pipeline_name}.json"
    page_md_path = page_root / f"{prefix}_text.md"
    page_marked_md_path = page_root / f"{prefix}_text_marked.md"

    page_json_path.write_text("[]", encoding="utf-8")
    res_json_path.write_text("{}",  encoding="utf-8")
    page_md_path.write_text("",     encoding="utf-8")
    page_marked_md_path.write_text("", encoding="utf-8")

    return {
        "num_blocks": 0,
        "markdown_chars": 0,
        "marked_markdown_chars": 0,
        "page_json_path": page_json_path,
        "res_json_path": res_json_path,
        "page_md_path": page_md_path,
        "page_marked_md_path": page_marked_md_path,
        "markdown_image_paths": [],
        "extra_img_paths": {},
    }



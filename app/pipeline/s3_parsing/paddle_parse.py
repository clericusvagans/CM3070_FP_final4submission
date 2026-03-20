# app/pipeline/s3_parsing/paddle_parse.py

import re
from dataclasses import dataclass
from typing import Any, Union

import numpy as np

from app.utils.logger import logger


@dataclass(frozen=True)
class PaddleParseConfig:
    pipeline: str = "pp_structure_v3"
    device: str = "gpu"
    lang: str | None = None  # actual per-page lang passed to parse()
    paddlex_config: Union[str, dict[str, Any]] | None = None


# ─── Low-level conversion helpers ───

def _as_list(x: Any) -> Any:
    """Recursively convert numpy arrays to plain Python lists."""
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (list, tuple)):
        return [_as_list(v) for v in x]
    if isinstance(x, dict):
        return {k: _as_list(v) for k, v in x.items()}
    return x


def _bbox_from_block_bbox(block_bbox: Any) -> list[int]:
    """
    Normalize PP-StructureV3 block_bbox to [x1, y1, x2, y2].

    PP-StructureV3 may return:
        shape (4,)   → [x1, y1, x2, y2] — returned directly.
        shape (4, 2) → polygon points    → bounding box computed from min/max.
    """
    bb = _as_list(block_bbox)

    if isinstance(bb, list) and len(bb) == 4 and all(isinstance(v, (int, float)) for v in bb):
        x1, y1, x2, y2 = bb
        return [int(x1), int(y1), int(x2), int(y2)]

    if isinstance(bb, list) and len(bb) == 4 and isinstance(bb[0], list) and len(bb[0]) == 2:
        xs = [p[0] for p in bb]
        ys = [p[1] for p in bb]
        return [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))]

    raise ValueError(f"Unrecognized block_bbox format: {type(block_bbox)} / {bb}")


def _normalize_label(label: str) -> str:
    """Normalize PP-Structure labels to the pipeline's canonical category strings."""
    if not label:
        return "Unknown"
    mapping = {
        "text": "Text", "title": "Title", "table": "Table",
        "figure": "Picture", "image": "Picture",
        "header": "Header", "footer": "Footer",
        "seal": "Seal", "stamp": "Seal",
        "formula": "Formula", "list": "List",
    }
    key = label.strip().lower()
    return mapping.get(key, key[:1].upper() + key[1:])


# ─── Markdown synthesis from blocks ───

def blocks_to_markdown(blocks: list[dict[str, Any]]) -> str:
    """
    Deterministic Markdown synthesis from parsed blocks.
    Use this when PP-StructureV3 does not return markdown_texts but blocks contain text.
    """
    def key(b: dict[str, Any]) -> tuple[int, int]:
        ro = b.get("reading_order")
        if isinstance(ro, int):
            return (0, ro)
        bbox = b.get("bbox") or [0, 0, 0, 0]
        return (1, int(bbox[1] or 0) * 100000 + int(bbox[0] or 0))

    out: list[str] = []
    for b in sorted(blocks, key=key):
        cat = str(b.get("category") or "").lower()
        txt = str(b.get("text") or "").strip()
        if not txt:
            continue
        if cat in {"title", "header"}:
            out.append(f"# {txt}\n")
        elif cat in {"text", "paragraph", "list"}:
            out.append(f"{txt}\n")
        elif cat == "table":
            out.append("\n<!-- TABLE: see page.json or crops -->\n")
        elif cat in {"picture", "figure", "image"}:
            out.append("\n<!-- IMAGE: see crops -->\n")
        else:
            out.append(f"{txt}\n")

    md = "\n".join(out).strip()
    return f"{md}\n" if md else ""


# ─── Marked markdown for unified translation ───

def _validate_block_order_consistency(blocks: list[dict[str, Any]]) -> bool:
    """
    Sanity check: verify block_order values don't conflict with block_id sequence.
    Returns False if PP-StructureV3 has assigned reading order that conflicts with
    block_id ordering (logged as a warning; block_id is used regardless).
    """
    pairs = [
        (b.get("block_id", 0), b.get("block_order"))
        for b in blocks
        if b.get("block_order") is not None
    ]
    if len(pairs) < 2:
        return True
    pairs.sort(key=lambda x: x[0])
    prev = pairs[0][1]
    for _, order in pairs[1:]:
        if order <= prev:
            return False
        prev = order
    return True


def _sort_key_for_block(block: dict[str, Any]) -> int:
    """
    Sort key using block_id (document order assigned by PP-StructureV3).

    block_order is not used here because it only covers a subset of block types
    (skipping images, footers, etc.), making block_id more reliable for full
    document ordering.
    """
    block_id = block.get("block_id")
    if block_id is not None:
        return int(block_id)
    bbox = block.get("block_bbox") or [0, 0, 0, 0]
    if isinstance(bbox, list) and len(bbox) >= 2:
        return int(bbox[1]) * 100000 + int(bbox[0])
    return 999999


def _generate_image_filename(bbox: list[int]) -> str:
    """Generate the image filename that PP-StructureV3 uses for image blocks."""
    if not bbox or len(bbox) < 4:
        return "imgs/unknown_image.jpg"
    return f"imgs/img_in_image_box_{bbox[0]}_{bbox[1]}_{bbox[2]}_{bbox[3]}.jpg"


def _estimate_image_width_percent(bbox: list[int], page_width: int = 2481) -> int:
    """
    Estimate display width percentage from bbox and page width.
    Default page_width of 2481 is typical for A4 at 300 DPI.
    """
    if not bbox or len(bbox) < 4 or page_width <= 0:
        return 50
    pct = int(((bbox[2] - bbox[0]) / page_width) * 100)
    return max(10, min(100, pct))


def generate_marked_markdown(
    parsing_res_list: list[dict[str, Any]],
    original_markdown: str = "",
    page_width: int = 2481,
) -> str:
    """
    Generate markdown with [#NNN] … [/#NNN] block markers for unified translation.

    Markers enable:
        1. Single-pass translation of the entire page (better context).
        2. Post-processing extraction of individual block translations.
        3. Preservation of document structure through translation.

    For image blocks, any extracted text (block_content) is injected as a <p>
    element within the image div so no OCR text is lost.

    Blocks are sorted by block_id (PP-StructureV3 document order). block_order
    is not used because it only covers a subset of block types. A consistency
    check warns if block_order conflicts with block_id ordering.

    Args:
        parsing_res_list: Block dicts from PP-StructureV3 (block_id, block_label,
                          block_content, block_bbox, block_order).
        original_markdown: Original markdown from PP-StructureV3 (used for table
                           HTML preservation).
        page_width: Page width in pixels for image width estimation.

    Returns:
        Markdown string with block markers.
    """
    if not parsing_res_list:
        return ""

    if not _validate_block_order_consistency(parsing_res_list):
        logger.warning(
            "[paddle_parse] block_order values conflict with block_id sequence; "
            "using block_id for ordering"
        )

    sorted_blocks = sorted(parsing_res_list, key=_sort_key_for_block)
    md_parts: list[str] = []

    for block in sorted_blocks:
        block_id  = block.get("block_id", 0)
        label = str(block.get("block_label") or "text").lower()
        content = str(block.get("block_content") or "").strip()
        bbox = block.get("block_bbox") or [0, 0, 0, 0]
        if isinstance(bbox, list) and len(bbox) >= 4:
            bbox = [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])]
        else:
            bbox = [0, 0, 0, 0]

        marker_id = f"{block_id:03d}"
        parts: list[str] = [f"[#{marker_id}]"]

        if label == "image":
            img_fn = _generate_image_filename(bbox)
            width_pct = _estimate_image_width_percent(bbox, page_width)
            if content:
                parts.append(
                    f'<div style="text-align: center;">'
                    f'<img src="{img_fn}" alt="Image" width="{width_pct}%" />'
                    f'<p>{content}</p>'
                    f'</div>'
                )
            else:
                parts.append(
                    f'<div style="text-align: center;">'
                    f'<img src="{img_fn}" alt="Image" width="{width_pct}%" />'
                    f'</div>'
                )

        elif label == "table":
            if content and "<table" in content.lower():
                parts.append(f'<div style="text-align: center;">{content}</div>')
            elif content:
                parts.append(content)
            else:
                parts.append("<!-- TABLE: no content extracted -->")

        elif label in ("title", "paragraph_title"):
            if content:
                parts.append(f"## {content}")

        else:   # text, paragraph, list, header, footer, footnote, etc.
            if content:
                parts.append(content)

        parts.append(f"[/#{marker_id}]")
        md_parts.append("\n".join(parts))

    return "\n\n".join(md_parts)


def extract_blocks_from_marked_markdown(marked_md: str) -> dict[int, str]:
    """
    Extract block texts from marker-delimited markdown.

    Parses [#NNN] … [/#NNN] markers and returns a dict mapping block_id to
    plain text (HTML tags stripped).

    Robustness note: the final block may occasionally lose its closing marker
    during MT generation (Gemma Translate can truncate the tail of the output).
    The pattern includes an EOF fallback to recover the last unterminated block.
    This is not a failproof safeguard — a generative model is unpredictable.
    """
    # Matches either a closed block [#NNN] ... [/#NNN] or a final unterminated
    # block that runs to EOF. The EOF fallback is intentionally narrow.
    pattern = r'\[#(\d{3})\]\s*(.*?)(?:\s*\[/#\1\]|\s*$)'
    blocks: dict[int, str] = {}

    for match in re.finditer(pattern, marked_md, re.DOTALL):
        block_id   = int(match.group(1))
        plain_text = strip_html_to_text(match.group(2).strip())
        if plain_text:
            blocks[block_id] = plain_text

    return blocks


def strip_html_to_text(html: str) -> str:
    """Strip HTML tags and markdown heading markers; normalize whitespace."""
    if not html:
        return ""
    text = re.sub(r'<[^>]+>', ' ', html)
    text = re.sub(r'^#+\s*', '', text, flags=re.MULTILINE)
    for entity, char in [('&nbsp;', ' '), ('&lt;', '<'), ('&gt;', '>'),
                         ('&amp;', '&'), ('&quot;', '"')]:
        text = text.replace(entity, char)
    return re.sub(r'\s+', ' ', text).strip()


def get_marked_markdown_for_translation(
    parsed: dict[str, Any],
    page_width: int = 2481,
) -> str:
    """
    Convenience entry point: generate marked markdown from PaddleParser.parse() output.

    Extracts parsing_res_list from parsed["raw_json"] and passes it to
    generate_marked_markdown() along with the original markdown text for
    table HTML preservation.
    """
    raw_json = parsed.get("raw_json") or {}
    payload = raw_json
    if isinstance(payload, dict):
        payload = payload.get("res", payload)

    parsing_res_list = payload.get("parsing_res_list") or [] if isinstance(payload, dict) else []

    md_dict = parsed.get("markdown") or {}
    original_markdown = md_dict.get("markdown_texts") or ""

    return generate_marked_markdown(
        parsing_res_list=parsing_res_list,
        original_markdown=original_markdown,
        page_width=page_width,
    )


# ─── PaddleParser ───

class PaddleParser:
    """
    Thin wrapper around PPStructureV3 with per-language pipeline caching.

    The singleton (paddle_singleton.py) holds one PaddleParser per worker process. Internally, PaddleParser caches a PPStructureV3 instance per language so repeated languages reuse the same in-memory model without re-initializing Paddle weights on every page.
    """

    def __init__(self, cfg: PaddleParseConfig):
        self.cfg        = cfg
        self._pipelines: dict[str, Any] = {}

    def _build_pipeline(self, lang: str | None) -> Any:
        if self.cfg.pipeline != "pp_structure_v3":
            raise ValueError("This adapter currently implements PPStructureV3 only.")

        from paddleocr import PPStructureV3

        effective_lang = (lang or self.cfg.lang or "en").strip() or "en"
        kwargs: dict[str, Any] = {
            "device": self.cfg.device or "gpu",
            "lang": effective_lang,
        }
        if self.cfg.paddlex_config is not None:
            kwargs["paddlex_config"] = self.cfg.paddlex_config
        return PPStructureV3(**kwargs)

    def _get_pipeline(self, lang: str | None) -> Any:
        effective_lang = (lang or self.cfg.lang or "en").strip() or "en"
        if effective_lang not in self._pipelines:
            self._pipelines[effective_lang] = self._build_pipeline(effective_lang)
        return self._pipelines[effective_lang]

    def parse(self, image_path: str, *, lang: str | None = None) -> dict[str, Any]:
        """
        Run PP-StructureV3 on a single page image and return normalized outputs.

        Args:
            image_path: Filesystem path to the rendered page image.
            lang:   Per-page language hint. Falls back to cfg.lang then "en".
                    A PP-StructureV3 pipeline for this language is cached
                    within the worker process and reused across pages.

        Returns:
            {
                "blocks":   [{bbox, category, text, reading_order, block_index}, …],
                "markdown": res.markdown dict (markdown_texts, markdown_images,
                            page_continuation_flags),
                "img":      res.img dict (layout_det_res, overall_ocr_res, …),
                "raw_json": dict  (res.json converted to plain Python),
            }

        Notes:
            - res.markdown and res.img are passed through as-is from Paddle.
              markdown_texts is a plain string, not a list.
            - blocks are derived from res.json parsing_res_list (normalized form).
        """
        pp = self._get_pipeline(lang)
        results = pp.predict(image_path)

        if not results:
            return {"blocks": [], "markdown": {}, "img": {}, "raw_json": {}}

        res0 = results[0]
        raw_json = _as_list(getattr(res0, "json", {}) or {})
        img_dict = getattr(res0, "img", None) or {}
        md_dict = getattr(res0, "markdown", None) or {}

        # Unwrap the common "res" envelope
        payload = raw_json
        if isinstance(payload, dict):
            payload = payload.get("res", payload)
            if isinstance(payload, dict) and isinstance(payload.get("res"), dict):
                payload = payload["res"]

        parsing_res_list = payload.get("parsing_res_list") or [] if isinstance(payload, dict) else []

        blocks: list[dict[str, Any]] = []
        for i, elem in enumerate(parsing_res_list):
            if not isinstance(elem, dict):
                continue
            bbox = _bbox_from_block_bbox(elem.get("block_bbox"))
            label = _normalize_label(str(elem.get("block_label") or "Unknown"))
            content = elem.get("block_content")
            block_order = elem.get("block_order")
            blocks.append({
                "block_index": i,
                "bbox": bbox,
                "category": label,
                "text": "" if content is None else str(content),
                "reading_order": int(block_order) if block_order is not None else None,
            })

        return {
            "blocks": blocks,
            "markdown": md_dict,  # res.markdown passed through intact
            "img": img_dict,  # res.img passed through intact
            "raw_json": raw_json,
        }

    def close(self) -> None:
        """
        Best-effort teardown to allow CUDA memory to be reclaimed.
        Drops references to internal Paddle predictor objects; enables
        GC + paddle.device.cuda.empty_cache() to free VRAM.
        """
        for attr in ("pipeline", "predictor", "model", "ocr", "engine"):
            if hasattr(self, attr):
                try:
                    setattr(self, attr, None)
                except Exception:
                    pass




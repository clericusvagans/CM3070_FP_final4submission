# app/pipeline/s4_nlp/mt_stage.py

"""
Machine Translation stage (S4-MT) for the standard DB-backed pipeline.

Translation strategy
--------------------
Two modes, tried in order per page:

1) Unified single-pass (default, recommended)
   Reads marked markdown from page_metadata["paddle"]["page_marked_md_uri"].
   Translates the entire page in ONE pass for better context, then
   post-processes to extract per-block translations.
   Populates both PageMT.text and BlockMT.text from the single translation.

2) Legacy two-pass (fallback)
   Used when marked markdown is unavailable.
   Translates Page.text_layer for PageMT.text; translates each
   PageBlock.text separately for BlockMT.text.

Marked markdown format
----------------------
    [#000]
    Block content here...
    [/#000]

    [#001]
    ## Next block with heading
    [/#001]

CT2 fallback note
-----------------
Some CT2 / OPUS models do not reliably preserve the [#NNN] markers through
translation. If the post-translation extractor recovers fewer blocks than
the source had, unified mode falls back to legacy mode for that page.

Artifacts written
-----------------
    <doc_root>/<page_number:04d>/pXXXX_text_mt.<langtag>.md

Database outputs
----------------
    PageMT rows  (lang, text, uri)
    BlockMT rows (lang, text)
"""

import re
import subprocess
import time
from datetime import datetime, timezone
from typing import Any

from sqlalchemy.orm import Session

from app.models import BlockMT, Document, Page, PageBlock, PageMT
from app.pipeline.s4_nlp.translate_base import (
    BaseTranslatorBackend,
    TextChunker,
    translate_with_chunking,
)
from app.pipeline.s3_parsing.paddle_parse import (
    extract_blocks_from_marked_markdown,
    strip_html_to_text,
)
from app.config.mt_config import (
    MT_LANG_MODEL_MAP,
    MT_HF_LANG_MODEL_MAP,
    MT_VLLM_LANG_MODEL_MAP,
    get_backend,
    get_batch_size,
    get_manual_token_limit_override,
    get_chars_per_token_override,
    get_chars_per_token_for_script,
    is_debug_chunks_enabled,
)
from app.utils import storage
from app.config.config import settings
from app.utils.logger import logger


# ─── Thermal management ───

GPU_TEMP_THRESHOLD      = 78    # Begin cooling pause above this temp (°C)
GPU_TEMP_CRITICAL       = 85    # Force longer pause above this temp (°C)
GPU_TEMP_CHECK_INTERVAL = 5     # Check every N pages
COOLING_PAUSE_NORMAL    = 2.0   # Seconds to pause when above threshold
COOLING_PAUSE_CRITICAL  = 10.0  # Seconds to pause when critical
FLUSH_INTERVAL          = 5     # Flush to DB every N pages (crash safety)


def get_gpu_temperature() -> int | None:
    """Query current GPU temperature via nvidia-smi. Returns °C or None if unavailable."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            temps = [int(t.strip()) for t in result.stdout.strip().split("\n") if t.strip()]
            return max(temps) if temps else None
    except (subprocess.TimeoutExpired, FileNotFoundError, ValueError):
        pass
    return None


def check_thermal_and_pause(page_index: int, force_check: bool = False) -> None:
    """
    Check GPU temperature and pause if needed to prevent thermal throttling.

    Checks only every GPU_TEMP_CHECK_INTERVAL pages unless force_check=True.
    """
    if not force_check and (page_index % GPU_TEMP_CHECK_INTERVAL != 0):
        return
    temp = get_gpu_temperature()
    if temp is None:
        return
    if temp >= GPU_TEMP_CRITICAL:
        logger.warning(
            f"[MT] GPU temperature critical: {temp}°C >= {GPU_TEMP_CRITICAL}°C, "
            f"pausing {COOLING_PAUSE_CRITICAL}s"
        )
        time.sleep(COOLING_PAUSE_CRITICAL)
    elif temp >= GPU_TEMP_THRESHOLD:
        logger.info(
            f"[MT] GPU temperature elevated: {temp}°C >= {GPU_TEMP_THRESHOLD}°C, "
            f"pausing {COOLING_PAUSE_NORMAL}s"
        )
        time.sleep(COOLING_PAUSE_NORMAL)


# ─── Backend cache ───

# cache_key → (backend_instance, label, model_name)
_backend_cache: dict[str, tuple[BaseTranslatorBackend, str, str]] = {}


def clear_backend_cache() -> None:
    """Clear the backend cache (useful for testing or configuration switches)."""
    _backend_cache.clear()


def release_mt_backend() -> None:
    """
    Release all cached MT backends and free GPU memory.

    Call this at the end of the MT stage if subsequent pipeline stages need
    GPU headroom (analogous to release_paddle_parser() in S3).

    Steps: close() each cached backend → clear cache → clear Gemma singleton
    → run GC (3 passes) → clear PyTorch CUDA cache.
    """
    global _backend_cache

    released = 0
    for cache_key, (backend, label, model_name) in list(_backend_cache.items()):
        try:
            if hasattr(backend, "close"):
                backend.close()
            released += 1
        except Exception as e:
            logger.warning(f"[MT] Error releasing backend {label}:{model_name}: {e}")
    _backend_cache.clear()

    try:
        from app.pipeline.s4_nlp.translate_gemma_backend import clear_translategemma_backend
        clear_translategemma_backend()
    except ImportError:
        pass
    except Exception as e:
        logger.warning(f"[MT] Error clearing Gemma singleton: {e}")

    try:
        import gc
        gc.collect(); gc.collect(); gc.collect()
    except Exception:
        pass

    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            alloc    = torch.cuda.memory_allocated() / 1024 ** 2
            reserved = torch.cuda.memory_reserved()  / 1024 ** 2
            logger.info(
                f"[MT] GPU memory after release: "
                f"allocated={alloc:.0f}MB reserved={reserved:.0f}MB"
            )
    except Exception as e:
        logger.warning(f"[MT] Error clearing CUDA cache: {e}")

    logger.info(f"[MT] Backend cache released ({released} backend(s))")


def _resolve_device_hf(raw: str | None) -> int | str | None:
    """Normalize MT_HF_DEVICE env value to device specifier."""
    if not raw or raw.strip().lower() == "auto":
        return None
    raw = raw.strip().lower()
    if raw == "cpu":
        return "cpu"
    try:
        return int(raw)
    except ValueError:
        return raw


# ─── Backend factory ───

def get_backend_for_lang(
    lang: str | None,
    script: str | None = None,
    target_lang: str = "en",
) -> tuple[BaseTranslatorBackend, str, str]:
    """
    Return (backend_instance, label, model_name) for the given language.

    Backends are cached per model key for the worker process lifetime.

    Args:
        lang:        Source language code (e.g. "sr", "uk").
        script:      Optional script code for chars-per-token estimation.
        target_lang: Target language code (used by the gemma backend).
    """
    backend_env = get_backend()
    lang_key = (lang or "").lower().split("-")[0]

    if backend_env == "ct2":
        model_key = (MT_LANG_MODEL_MAP.get(lang_key) or MT_LANG_MODEL_MAP["__default__"])
    elif backend_env == "hf":
        model_key = (MT_HF_LANG_MODEL_MAP.get(lang_key) or MT_HF_LANG_MODEL_MAP["__default__"])
    elif backend_env == "vllm":
        model_key = (MT_VLLM_LANG_MODEL_MAP.get(lang_key) or MT_VLLM_LANG_MODEL_MAP["__default__"])
    elif backend_env == "gemma":
        model_key = settings.MT_GEMMA_MODEL
    else:
        raise ValueError(
            f"Unknown MT_BACKEND={backend_env!r}. "
            "Supported: 'ct2', 'hf', 'vllm', 'gemma'."
        )

    cache_key = (
        f"{backend_env}:{model_key}:{lang_key}" if backend_env == "gemma"
        else f"{backend_env}:{model_key}"
    )

    if cache_key in _backend_cache:
        logger.debug(f"[MT] Cache hit: cache_key={cache_key!r} lang={lang!r}")
        cached = _backend_cache[cache_key]
        if backend_env == "gemma":
            cached[0].set_languages(lang_key, target_lang)
        return cached

    manual_max_tokens = get_manual_token_limit_override()
    chars_override = get_chars_per_token_override()
    if chars_override is None and script:
        chars_override = get_chars_per_token_for_script(script)

    if backend_env == "ct2":
        from app.pipeline.s4_nlp.translate_CT2_backend import CT2TranslatorBackend

        model_dir = f"{str(settings.MT_CT2_MODEL_BASE_DIR).rstrip('/')}/{model_key}"
        logger.info(
            f"[MT] Loading CT2 backend lang={lang!r} model_dir={model_dir!r} "
            f"device={settings.MT_CT2_DEVICE!r} compute_type={settings.MT_CT2_COMPUTE!r}"
        )
        instance = CT2TranslatorBackend(
            model_dir=model_dir,
            device=str(settings.MT_CT2_DEVICE).strip(),
            compute_type=str(settings.MT_CT2_COMPUTE).strip(),
            max_tokens_override=manual_max_tokens,
            chars_per_token_override=chars_override,
        )
        label = "ct2"
        model_name = model_dir

    elif backend_env == "gemma":
        from app.pipeline.s4_nlp.translate_gemma_backend import TranslateGemmaBackend

        logger.info(
            f"[MT] Loading TranslateGemma backend lang={lang!r} → {target_lang!r} "
            f"model={model_key!r}"
        )
        instance = TranslateGemmaBackend(
            model_name=model_key,
            source_lang=lang_key,
            target_lang=target_lang,
            device_map=settings.MT_GEMMA_DEVICE_MAP,
            load_in_4bit=settings.MT_GEMMA_LOAD_IN_4BIT,
            load_in_8bit=settings.MT_GEMMA_LOAD_IN_8BIT,
            max_new_tokens=settings.MT_GEMMA_MAX_NEW_TOKENS,
            max_tokens_override=manual_max_tokens,
            chars_per_token_override=chars_override,
        )
        label      = "gemma"
        model_name = model_key

    else:
        raise ValueError(f"Backend loading not implemented for {backend_env!r}")

    _backend_cache[cache_key] = (instance, label, model_name)
    logger.info(f"[MT] Backend ready: label={label!r} model={model_name!r}")
    return _backend_cache[cache_key]


# ─── Translation with optional validation ───

def _translate_text(
    backend: BaseTranslatorBackend,
    text: str,
    batch_size: int,
    validate_chunks: bool = False,
) -> str:
    """
    Translate text with optional chunk validation (DEBUG_MT_CHUNKS=1).

    Validation logs a warning for each chunk that exceeds the token limit;
    it does not abort translation.
    """
    if not text or not text.strip():
        return ""

    max_chars = backend.get_max_chars()
    chunks    = TextChunker(max_chars=max_chars).chunk(text)
    if not chunks:
        return ""

    if validate_chunks and hasattr(backend, "validate_chunk"):
        for i, chunk in enumerate(chunks):
            if not backend.validate_chunk(chunk):
                actual = backend.count_tokens(chunk) if hasattr(backend, "count_tokens") else "?"
                logger.warning(
                    f"[MT] Chunk {i} exceeds token limit: {len(chunk)} chars "
                    f"~{actual} tokens max_chars={max_chars}"
                )

    return translate_with_chunking(backend, text, batch_size, max_chars)


# ─── Unified translation helpers ───

def _strip_block_markers(marked_md: str) -> str:
    """Remove [#NNN] and [/#NNN] markers while preserving content."""
    if not marked_md:
        return ""
    text = re.sub(r'\[#\d{3}\]\s*', '', marked_md)
    text = re.sub(r'\s*\[/#\d{3}\]', '', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def _get_expected_block_count_from_source(marked_md: str) -> int:
    """
    Count the non-empty blocks in source marked markdown using the same extraction logic used after translation. Then compare extracted count to the number of blocks the extractor would find in the original, not to the raw marker count (remember: some source blocks are legitimately empty and produce no BlockMT row).
    """
    return len(extract_blocks_from_marked_markdown(marked_md)) if marked_md else 0


def _get_marked_markdown_for_page(page: Page) -> str | None:
    """Read the marked markdown file for a page from page_metadata["paddle"]["page_marked_md_uri"]."""
    marked_md_uri = ((page.page_metadata or {}).get("paddle") or {}).get("page_marked_md_uri")
    if not marked_md_uri:
        return None
    try:
        path = storage.resolve_local_path(marked_md_uri)
        if path.exists():
            return path.read_text(encoding="utf-8")
    except Exception as e:
        logger.warning(f"[MT] Failed to read marked markdown for page_id={page.id}: {e}")
    return None


def _build_block_id_to_db_id_map(db: Session, page_id: int) -> dict[int, int]:
    """
    Build a mapping from block_index (used in [#NNN] markers) to PageBlock.id.

    PP-StructureV3 block_id becomes block_index in PageBlock; the DB primary key is needed for BlockMT foreign keys.
    """
    blocks = (
        db.query(PageBlock.id, PageBlock.block_index)
        .filter(PageBlock.page_id == page_id)
        .all()
    )
    return {int(b.block_index): int(b.id) for b in blocks}


# ─── Main stage entrypoint ───

def translate_and_persist_document(
    db: Session,
    *,
    doc: Document,
    pages: list[Page],
    batch_size: int | None = None,
    strict: bool = False,
    target_lang: str = "en",
    validate_chunks: bool | None = None,
) -> dict[str, Any]:
    """
    Translate pages and persist PageMT / BlockMT rows.

    For each page:
        1. Try unified single-pass translation using marked markdown.
        2. Fall back to legacy two-pass if marked markdown is unavailable.

    DB writes are flushed every FLUSH_INTERVAL pages for crash safety and
    once more at the end.

    Args:
        db:              SQLAlchemy session.
        doc:             Document ORM instance.
        pages:           Pages to translate.
        batch_size:      Chunks per translation batch (default from settings).
        strict:          If True, raise on any page failure.
        target_lang:     Target language code (default "en").
        validate_chunks: Validate chunks against token limits (default from DEBUG_MT_CHUNKS).

    Returns:
        Summary dict with translation statistics.
    """
    backend_env = get_backend()

    if not pages:
        logger.warning(f"[MT] doc_id={doc.id} has no pages; skipping translation stage")
        return {
            "doc_id": doc.id, "pages": 0, "translated_pages": 0,
            "translated_blocks": 0, "skipped_pages": 0, "failed_pages": 0,
            "backend": backend_env, "models": set(),
        }

    if batch_size is None:
        batch_size = get_batch_size()
    if validate_chunks is None:
        validate_chunks = is_debug_chunks_enabled()

    target_tag = "eng"
    skipped_pages = 0
    failed_pages = 0
    translated_pages = 0
    translated_blocks = 0
    models_used: set[str] = set()

    for page_idx, page in enumerate(pages):
        check_thermal_and_pause(page_idx)

        raw_lang = page.primary_lang or doc.primary_lang or ""
        lang_key = raw_lang.lower().split("-")[0] or "__default__"
        script = page.primary_script or doc.primary_script

        if lang_key == target_lang:
            pm = dict(page.page_metadata or {})
            pm["mt"] = {"translate_ok": False, "error": "source_is_target_language",
                        "lang": lang_key, "target_lang": target_lang}
            page.page_metadata = pm
            skipped_pages += 1
            continue

        marked_md = _get_marked_markdown_for_page(page)
        kwargs = dict(
            db=db, page=page, doc=doc,
            lang_key=lang_key, script=script,
            target_lang=target_lang, target_tag=target_tag,
            batch_size=batch_size, validate_chunks=validate_chunks,
        )

        if marked_md and marked_md.strip():
            try:
                result = _translate_page_unified(marked_md=marked_md, **kwargs)
            except Exception as e:
                logger.error(f"[MT] Unified translation failed doc_id={doc.id} page_id={page.id}: {e}")
                if strict: raise
                failed_pages += 1
                continue
        else:
            src_text = (page.text_layer or "").strip()
            if not src_text:
                pm = dict(page.page_metadata or {})
                pm["mt"] = {"translate_ok": False, "error": "no_text_layer_or_marked_md",
                            "target_lang": target_lang}
                page.page_metadata = pm
                skipped_pages += 1
                continue
            try:
                result = _translate_page_legacy(src_page_text=src_text, **kwargs)
            except Exception as e:
                logger.error(f"[MT] Legacy translation failed doc_id={doc.id} page_id={page.id}: {e}")
                if strict: raise
                failed_pages += 1
                continue

        if result["success"]:
            translated_pages  += 1
            translated_blocks += result["blocks_translated"]
            models_used.add(result["model"])
        else:
            failed_pages += 1

        if (page_idx + 1) % FLUSH_INTERVAL == 0:
            db.flush()
            logger.debug(f"[MT] Periodic flush after {page_idx + 1} pages")

    db.flush()

    logger.info(
        f"[MT] doc_id={doc.id} pages={len(pages)} translated_pages={translated_pages} "
        f"translated_blocks={translated_blocks} skipped_pages={skipped_pages} "
        f"failed_pages={failed_pages} backend={backend_env} models={models_used}"
    )

    if backend_env == "gemma":
        try:
            release_mt_backend()
        except Exception as e:
            logger.warning(f"[MT] Failed to release backend doc_id={doc.id}: {e}")

    return {
        "doc_id": doc.id, "pages": len(pages),
        "translated_pages": translated_pages, "translated_blocks": translated_blocks,
        "skipped_pages": skipped_pages, "failed_pages": failed_pages,
        "backend": backend_env, "models": models_used,
        "model": next(iter(models_used), None),
    }


# ─── Per-page translation helpers ───

def _translate_page_unified(
    *,
    db: Session,
    page: Page,
    doc: Document,
    marked_md: str,
    lang_key: str,
    script: str | None,
    target_lang: str,
    target_tag: str,
    batch_size: int,
    validate_chunks: bool,
) -> dict[str, Any]:
    """
    Translate a page in a single pass using marked markdown.

    Strips markers for clean PageMT.text; extracts per-block translations
    for BlockMT rows.

    CT2 fallback: if the extractor recovers fewer blocks than expected
    (markers corrupted during translation), falls back to legacy mode.

    Returns:
        {"success": bool, "blocks_translated": int, "model": str}
    """
    page_id = int(page.id)
    page_number = int(page.page_number)

    backend, backend_label, backend_model = get_backend_for_lang(
        lang_key, script, target_lang=target_lang
    )
    model_key = backend_model or backend_label

    translated_marked_md = (_translate_text(backend, marked_md, batch_size, validate_chunks) or "").strip()

    if not translated_marked_md:
        logger.warning(f"[MT] Empty translation for page_id={page_id}")
        pm = dict(page.page_metadata or {})
        pm["mt"] = {"translate_ok": False, "error": "empty_translation", "target_lang": target_lang}
        page.page_metadata = pm
        return {"success": False, "blocks_translated": 0, "model": model_key}

    clean_mt_text = _strip_block_markers(translated_marked_md)
    block_translations = extract_blocks_from_marked_markdown(translated_marked_md)
    expected_count = _get_expected_block_count_from_source(marked_md)
    extracted_count = len(block_translations)

    logger.info(
        f"[MT] page_id={page_id} | Extracted/Expected blocks: "
        f"{extracted_count}/{expected_count} | "
        f"extracted block ids={sorted(block_translations.keys())}"
    )

    if extracted_count != expected_count:
        logger.warning(
            f"[MT] page_id={page_id} block count mismatch after unified translation: "
            f"expected={expected_count} extracted={extracted_count} "
            f"backend={backend_label} model={backend_model}"
        )

    # CT2 fallback: markers not reliably preserved → fall back to legacy mode.
    if backend_label == "ct2" and extracted_count != expected_count:
        src_text = (page.text_layer or "").strip()
        if src_text:
            logger.warning(
                f"[MT] CT2 unified translation recovered {extracted_count}/"
                f"{expected_count} blocks; falling back to legacy translation "
                f"for doc_id={doc.id} page_id={page_id}"
            )
            return _translate_page_legacy(
                db=db, page=page, doc=doc, src_page_text=src_text,
                lang_key=lang_key, script=script,
                target_lang=target_lang, target_tag=target_tag,
                batch_size=batch_size, validate_chunks=validate_chunks,
            )

    block_id_map = _build_block_id_to_db_id_map(db, page_id)

    db.query(PageMT).filter(
        PageMT.page_id == page_id, PageMT.lang == target_lang,
    ).delete(synchronize_session=False)

    prefix = storage.page_prefix(page_number)
    out_path = storage.page_dir(doc.id, page_number) / f"{prefix}_text_mt.{target_tag}.md"
    out_path.write_text(f"{clean_mt_text}\n", encoding="utf-8")
    uri = storage.to_output_uri(out_path)
    ts = datetime.now(timezone.utc)
    model_used = f"{backend_label}:{model_key}"

    db.add(PageMT(
        page_id=page_id, lang=target_lang, text=clean_mt_text,
        uri=uri, translated_at=ts, model=model_used,
    ))

    blocks_translated = 0
    for block_index, block_text in block_translations.items():
        db_block_id = block_id_map.get(block_index)
        if db_block_id is None:
            continue   # block in marked MD but not in DB (e.g. footer excluded from PageBlock)
        db.query(BlockMT).filter(
            BlockMT.block_id == db_block_id, BlockMT.lang == target_lang,
        ).delete(synchronize_session=False)
        db.add(BlockMT(
            block_id=db_block_id, lang=target_lang, text=block_text,
            translated_at=ts, model=model_used,
        ))
        blocks_translated += 1

    pm = dict(page.page_metadata or {})
    pm["mt"] = {
        "translate_ok": True, "target_lang": target_lang,
        "backend": backend_label, "model": backend_model,
        "page_mt_uri": uri, "chars_in": len(marked_md),
        "chars_out": len(clean_mt_text),
        "blocks_translated": blocks_translated, "mode": "unified",
    }
    page.page_metadata = pm
    return {"success": True, "blocks_translated": blocks_translated, "model": model_key}


def _translate_page_legacy(
    *,
    db: Session,
    page: Page,
    doc: Document,
    src_page_text: str,
    lang_key: str,
    script: str | None,
    target_lang: str,
    target_tag: str,
    batch_size: int,
    validate_chunks: bool,
) -> dict[str, Any]:
    """
    Translate a page using the legacy two-pass approach (fallback).

    Pass 1: translate Page.text_layer → PageMT.text.
    Pass 2: translate each PageBlock.text → BlockMT.text.

    Returns:
        {"success": bool, "blocks_translated": int, "model": str}
    """
    page_id = int(page.id)
    page_number = int(page.page_number)

    backend, backend_label, backend_model = get_backend_for_lang(
        lang_key, script, target_lang=target_lang
    )
    model_key = backend_model or backend_label

    mt_text = (_translate_text(backend, src_page_text, batch_size, validate_chunks) or "").strip()

    if not mt_text:
        logger.warning(f"[MT] Empty page translation for page_id={page_id}")
        pm = dict(page.page_metadata or {})
        pm["mt"] = {"translate_ok": False, "error": "empty_translation",
                    "target_lang": target_lang, "mode": "legacy"}
        page.page_metadata = pm
        return {"success": False, "blocks_translated": 0, "model": model_key}

    db.query(PageMT).filter(
        PageMT.page_id == page_id, PageMT.lang == target_lang,
    ).delete(synchronize_session=False)

    prefix = storage.page_prefix(page_number)
    out_path = storage.page_dir(doc.id, page_number) / f"{prefix}_text_mt.{target_tag}.md"
    out_path.write_text(f"{mt_text}\n", encoding="utf-8")
    uri = storage.to_output_uri(out_path)
    ts = datetime.now(timezone.utc)
    model_used = f"{backend_label}:{model_key}"

    db.add(PageMT(
        page_id=page_id, lang=target_lang, text=mt_text,
        uri=uri, translated_at=ts, model=model_used,
    ))

    blocks = (
        db.query(PageBlock)
        .filter(PageBlock.page_id == page_id)
        .order_by(PageBlock.block_index.asc())
        .all()
    )

    blocks_translated = 0
    for blk in blocks:
        txt = (blk.text or "").strip()
        if not txt:
            continue
        try:
            block_mt = (_translate_text(backend, txt, batch_size, validate_chunks) or "").strip()
            if not block_mt:
                continue
            db.query(BlockMT).filter(
                BlockMT.block_id == blk.id, BlockMT.lang == target_lang,
            ).delete(synchronize_session=False)
            db.add(BlockMT(
                block_id=blk.id, lang=target_lang, text=block_mt,
                translated_at=ts, model=model_used,
            ))
            blocks_translated += 1
        except Exception as e:
            logger.warning(f"[MT] Block translation failed block_id={blk.id}: {e}")

    pm = dict(page.page_metadata or {})
    pm["mt"] = {
        "translate_ok": True, "target_lang": target_lang,
        "backend": backend_label, "model": backend_model,
        "page_mt_uri": uri, "chars_in": len(src_page_text),
        "chars_out": len(mt_text),
        "blocks_translated": blocks_translated, "mode": "legacy",
    }
    page.page_metadata = pm
    return {"success": True, "blocks_translated": blocks_translated, "model": model_key}



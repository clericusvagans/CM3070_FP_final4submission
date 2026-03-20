# app/pipeline/s2_detection/lang_script_detect.py

"""
Stage S2: Language and script detection.

For each page:
    - Loads the rendered page image from page_metadata["image_uri"].
    - Runs Tesseract OSD to detect the dominant script (ISO 15924 code).
    - Runs Tesseract OCR to extract a text sample.
    - Uses Lingua to compute per-language confidence scores over that sample.

Per-page writes
---------------
    Page.lang_probs       {lang_code: confidence} dict from Lingua.
    Page.primary_lang     Top-scoring language code (or None).
    Page.primary_script   Detected script code, e.g. "Cyrl", "Latn" (or None).

    NOTE: ocr_lang_hint was removed from Page. All pages share the same candidate
    set, which is stored once in Document.declared_langs.

Document-level writes
---------------------
    Document.declared_langs   Sorted candidate list from settings.CANDIDATE_LANGS.
    Document.detected_langs   {lang_code: normalised_weight} aggregated across pages.
    Document.primary_lang     Top language by aggregated weight.
    Document.primary_script   Most frequent page-level script.

Candidate languages are read from ``settings.CANDIDATE_LANGS`` (config/config.py).
"""

import re
from collections import Counter
from dataclasses import dataclass
from typing import Any

import cv2
from PIL import Image
import pytesseract
from pytesseract import Output
from sqlalchemy.orm import Session
from lingua import LanguageDetectorBuilder

from app.config.config import settings
from app.models import Document, Page
from app.utils.storage import resolve_local_path
from app.utils.logger import logger


# ─── Language / script mappings ───

def _lingua_lang_map() -> dict[str, Any]:
    from lingua import Language
    return {
        "en": Language.ENGLISH,
        "fr": Language.FRENCH,
        "nl": Language.DUTCH,
        "ru": Language.RUSSIAN,
        "uk": Language.UKRAINIAN,
        "de": Language.GERMAN,
        "es": Language.SPANISH,
        "it": Language.ITALIAN,
        "pl": Language.POLISH,
    }


_TESS_2L_TO_3L = {
    "de": "deu", "es": "spa", "it": "ita",
    "pl": "pol", "en": "eng", "fr": "fra",
    "nl": "nld", "ru": "rus", "uk": "ukr",
}

_SCRIPT_TO_ISO15924 = {
    "latin": "Latn", "cyrillic": "Cyrl", "arabic": "Arab",
    "hebrew": "Hebr", "greek": "Grek", "devanagari": "Deva",
    "han": "Hani",   "hangul": "Hang",
}


# ─── Internal helpers ───

def _extract_osd(img: Image.Image) -> dict:
    """Run Tesseract OSD; never raise — log and return defaults on failure."""
    try:
        return pytesseract.image_to_osd(img, output_type=Output.DICT)
    except Exception as e:
        logger.warning(f"[detect_langs_and_scripts] image_to_osd failed: {e}")
        return {
            "orientation": 0, "rotate": 0, "orientation_confidence": 0.0,
            "script": None, "script_confidence": 0.0,
        }


def _tesseract_lang_arg(candidate_langs: list[str]) -> str:
    """Build the Tesseract `lang` argument from ISO 639-1 candidate codes."""
    langs = [_TESS_2L_TO_3L.get(c.lower().strip(), c.lower().strip()) for c in candidate_langs]
    return "+".join(dict.fromkeys(langs)) or "eng"


def _load_page_image(page: Page) -> Image.Image:
    """Load the rasterized page image from page_metadata["image_uri"] as a PIL RGB Image."""
    meta = page.page_metadata or {}
    uri = meta.get("image_uri")
    if not uri:
        raise RuntimeError(f"Page {page.id} missing image_uri in metadata")
    path = resolve_local_path(uri)
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"Cannot read page image: {path}")
    return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


def _build_lingua_detector(candidate_langs: list[str]):
    """Build a Lingua detector restricted to the given ISO 639-1 candidate codes."""
    lang_map = _lingua_lang_map()
    langs = [lang_map[c] for c in candidate_langs if c in lang_map] or [lang_map["en"]]
    return LanguageDetectorBuilder.from_languages(*langs).build()


def _lingua_probs(detector, text: str) -> dict[str, float]:
    """
    Compute per-language confidence scores for `text` using Lingua.
    Returns an empty dict if text is too short (<20 chars) or Lingua fails.
    Scores are rounded to 6 decimal places for compact JSONB storage.
    """
    if not text or len(text) < 20:
        return {}
    try:
        results = detector.compute_language_confidence_values(text)
    except Exception:
        logger.warning("[detect_langs_and_scripts] Lingua confidence computation failed")
        return {}
    probs = {}
    for r in results:
        iso = getattr(r.language, "iso_code_639_1", None)
        code = str(iso.name).lower() if iso else None
        if code:
            probs[code] = round(float(r.value), 6)
    return probs


def _detect_script(img: Image.Image) -> str | None:
    """
    Return an ISO 15924-ish script code from Tesseract OSD, or the raw script name
    if unmapped, or None if unavailable. Matches the "best effort, never throw" approach.
    """
    osd = _extract_osd(img)
    raw = (osd.get("script") or "").strip()
    if not raw:
        return None
    return _SCRIPT_TO_ISO15924.get(raw.lower()) or raw


# ─── Public API ───

@dataclass
class LangScriptSummary:
    pages:              int
    pages_with_lang:    int
    pages_with_script:  int
    doc_primary_lang:   str | None
    doc_primary_script: str | None


def detect_langs_and_scripts(
    db: Session,
    *,
    doc: Document,
    pages: list[Page],
) -> LangScriptSummary:
    """
    S2: detect language and script for each page; aggregate at document level.

    Candidate languages come from settings.CANDIDATE_LANGS — the single source of truth. The list is stored on Document.declared_langs so downstream stages (S3 Paddle, S4 NLP) can read it without reloading config.

    Per-page loop is fully exception-safe: any crash on a single page (Tesseract OSD, cv2, PIL, Lingua) is logged with a full traceback and the page is skipped so the rest of the document is not affected.

    JSONB copy-on-write: page.lang_probs and doc.detected_langs are assigned as new
    dict objects so SQLAlchemy's change tracking fires correctly.
    """
    candidate_langs = [c.lower().strip() for c in (settings.CANDIDATE_LANGS or []) if c.strip()] or ["en"]

    logger.info(
        f"[Orchestrator][S2] STARTING: doc_id={doc.id} "
        f"pages={len(pages)} candidate_langs={candidate_langs}"
    )

    # Building the detector and Tesseract arg can fail loudly if settings.CANDIDATE_LANGS
    # contains unrecognised codes — better than silent wrong behaviour downstream.
    detector  = _build_lingua_detector(candidate_langs)
    tess_lang = _tesseract_lang_arg(candidate_langs)

    script_votes: list[str] = []
    acc_probs: Counter = Counter()
    pages_with_lang: int = 0
    pages_with_script: int = 0

    for p in pages:
        # Wrap the entire per-page block: a crash on any one page (OSD, OCR, Lingua)
        # must not propagate out of the loop and abort the whole stage.
        try:
            try:
                img = _load_page_image(p)
            except Exception as e:
                logger.warning(
                    f"[detect_langs_and_scripts] page_id={p.id} "
                    f"page_number={p.page_number}: cannot load image: {e}"
                )
                continue

            try:
                script = _detect_script(img)
            except Exception as e:
                logger.warning(
                    f"[detect_langs_and_scripts] page_id={p.id} "
                    f"page_number={p.page_number}: OSD failed: {e}"
                )
                script = None

            if script:
                p.primary_script = script
                script_votes.append(script)
                pages_with_script += 1
            else:
                logger.info(f"[detect_langs_and_scripts] page_id={p.id} script=None")

            try:
                text = pytesseract.image_to_string(img, lang=tess_lang, config="--psm 6 --oem 1")
            except Exception as e:
                logger.warning(
                    f"[detect_langs_and_scripts] page_id={p.id} "
                    f"page_number={p.page_number}: OCR failed: {e}"
                )
                text = ""

            text  = re.sub(r"\s+", " ", (text or "").strip())
            probs = _lingua_probs(detector, text)

            p.lang_probs = dict(probs) if probs else {}

            if probs:
                pages_with_lang  += 1
                p.primary_lang    = max(probs.items(), key=lambda kv: kv[1])[0]
                acc_probs.update(probs)
            else:
                logger.debug(
                    f"[detect_langs_and_scripts] page_id={p.id} "
                    "no lang detected (text too short or Lingua inconclusive)"
                )

            db.add(p)

        except Exception as e:
            logger.exception(
                f"[detect_langs_and_scripts] page_id={p.id} "
                f"page_number={p.page_number}: unexpected error, skipping: {e}"
            )

    # ─── Document-level aggregation ───
    doc.declared_langs = sorted(candidate_langs)

    detected: dict[str, float] = {}
    if acc_probs:
        total = float(sum(acc_probs.values()))
        if total > 0:
            detected = {k: round(float(v) / total, 6) for k, v in acc_probs.items()}

    doc.detected_langs  = dict(detected)
    doc.primary_lang = max(detected.items(), key=lambda kv: kv[1])[0] if detected else None
    doc.primary_script = Counter(script_votes).most_common(1)[0][0] if script_votes else None

    logger.info(
        f"[detect_langs_and_scripts] doc_id={doc.id} done: "
        f"primary_lang={doc.primary_lang} primary_script={doc.primary_script} "
        f"pages_with_lang={pages_with_lang}/{len(pages)} "
        f"pages_with_script={pages_with_script}/{len(pages)} "
        f"declared={doc.declared_langs} detected={detected}"
    )

    db.add(doc)
    db.flush()

    return LangScriptSummary(
        pages=len(pages),
        pages_with_lang=pages_with_lang,
        pages_with_script=pages_with_script,
        doc_primary_lang=doc.primary_lang,
        doc_primary_script=doc.primary_script,
    )




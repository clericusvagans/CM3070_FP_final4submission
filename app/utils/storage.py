# app/utils/storage.py

"""
Unified storage utilities for the document-processing pipeline.

Filesystem layout
-----------------
1) Persistent output tree (under config.OUTPUT_ROOT)

       <OUTPUT_ROOT>/documents/000000123/
           SOURCEFILE/                          original uploaded artifact
           0001/                                page folders (1-based, zero-padded)
               p0001_source.png
               p0001_text.md
               p0001_text_mt.eng.md
               imgs/...
               LAYOUT/
                   p0001_img_layout_order_res.png
                   ...
           0002/
               ...

2) Ephemeral worker scratch space (also under config.OUTPUT_ROOT)

       <OUTPUT_ROOT>/_scratch/job_<uuid>_*/

DB-free (test) runs
-------------------
Tests use a document root under OUTPUT_TEMP with a timestamp-keyed folder name
instead of an integer doc_id. The on-disk shape is identical:

    <OUTPUT_TEMP>/documents/<YYYYMMDDTHHMMSS_filename>/
        SOURCEFILE/
        0001/  0002/  ...

URI scheme
----------
All stored paths use an `output://` URI:
    output:///documents/000000001/SOURCEFILE/file.pdf   (triple-slash canonical form)
    output://documents/000000001/SOURCEFILE/file.pdf    (two-slash; handled by from_output_uri)

`from_output_uri` correctly handles both forms: when urlparse() sees a
two-slash URI it puts the first path segment into `netloc`; the function
reconstructs the full relative path by joining netloc + path.
"""

import json
import os
from pathlib import Path
from typing import Any
from urllib.parse import urlparse, unquote

import cv2
import numpy as np

from app.config.config import OUTPUT_ROOT, OUTPUT_TEMP
from app.utils.logger import logger


SOURCEFILE_DIRNAME = "SOURCEFILE"
LAYOUT_DIRNAME = "LAYOUT"
SUPPORTED_EXTS = {".pdf", ".tif", ".tiff", ".png", ".jpg", ".jpeg"}
OUTPUT_SCHEME = "output"


# ─── Root directory helpers ───

def resolve_and_mkdir_output(dir_type: Path | str | None = None) -> Path:
    """
    Resolve the output root to an absolute Path and ensure it exists.
    Defaults to OUTPUT_ROOT when dir_type is None (read at call time, not import time).
    """
    root = Path(OUTPUT_ROOT if dir_type is None else dir_type).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    return root


def mkdir_and_return_documents_path(dir_type: Path | str | None = None) -> Path:
    """Return `<OUTPUT_ROOT>/documents`, creating it if needed."""
    root = resolve_and_mkdir_output(dir_type) / "documents"
    root.mkdir(parents=True, exist_ok=True)
    return root


def mkdir_and_return_scratch_path(dir_type: Path | str | None = None) -> Path:
    """Return `<OUTPUT_ROOT>/_scratch`, creating it if needed."""
    root = resolve_and_mkdir_output(dir_type) / "_scratch"
    root.mkdir(parents=True, exist_ok=True)
    return root


# def test_run_root(run_id: str, *, dir_type: Path | str | None = None) -> Path:
#     """
#     Legacy helper: root folder for a DB-free test ingest run.

#         <OUTPUT_TEMP>/runs/<run_id>/

#     Prefer `test_doc_root()` for new code, which uses the aligned per-page
#     folder structure (SOURCEFILE/ + 0001/, 0002/, …).
#     """
#     root = resolve_and_mkdir_output(dir_type or OUTPUT_TEMP) / "runs" / run_id
#     root.mkdir(parents=True, exist_ok=True)
#     return root


# ─── Internal safety helpers ───

def _resolve_under_output_root(rel: str | Path, dir_type: Path | str | None = None) -> Path:
    """
    Anti-traversal guard: resolve `rel` under OUTPUT_ROOT.
    Raises ValueError if the resolved path escapes the root.
    """
    root = resolve_and_mkdir_output(dir_type)
    candidate = (root / Path(rel)).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as e:
        raise ValueError(
            f"Resolved path escapes OUTPUT_ROOT: rel={rel!r} -> {candidate}"
        ) from e
    return candidate


# ─── Relative / absolute path helpers ───

def to_rel_output_path(p: Path, dir_type: Path | str | None = None) -> str:
    """
    Return a POSIX-style path string relative to OUTPUT_ROOT.
    `expanduser()` is defensive hardening for paths containing `~`.
    """
    root = resolve_and_mkdir_output(dir_type)
    rp = p.expanduser().resolve()
    try:
        return rp.relative_to(root).as_posix()
    except ValueError as e:
        raise ValueError(f"Path is not under OUTPUT_ROOT: path={rp} root={root}") from e


def from_rel_output_path(rel: str, dir_type: Path | str | None = None) -> Path:
    """Resolve a DB-stored relative output path to an absolute Path (traversal-safe)."""
    return _resolve_under_output_root(rel, dir_type=dir_type)


# ─── URI helpers ───

def to_output_uri(abs_path: Path, dir_type: Path | str | None = None) -> str:
    """
    Convert an absolute path under OUTPUT_ROOT into an `output://` URI.

    Canonical form uses triple slash so the full relative path stays in `path`:
        output:///documents/000000001/SOURCEFILE/file.pdf
    """
    root = resolve_and_mkdir_output(dir_type)
    rel  = abs_path.resolve().relative_to(root).as_posix()
    return f"{OUTPUT_SCHEME}:///{rel}"


def from_output_uri(uri: str, dir_type: Path | str | None = None) -> Path:
    """
    Resolve an `output://` URI to an absolute local path under OUTPUT_ROOT.

    Handles both canonical (triple-slash) and two-slash forms. With a two-slash
    URI, `urlparse()` puts the first path segment into `netloc`; we
    reconstruct the full relative path by joining `netloc + path`.
    """
    u = urlparse(uri)
    if u.scheme != OUTPUT_SCHEME:
        raise ValueError(f"Not an output:// URI: {uri}")

    netloc = unquote(u.netloc or "").strip()
    path = unquote(u.path   or "").lstrip("/")

    rel = f"{netloc}/{path}" if netloc and path else (netloc or path)
    if not rel:
        raise ValueError(f"Malformed output:// URI (empty path): {uri}")

    return _resolve_under_output_root(rel, dir_type=dir_type)


def resolve_local_path(path_or_uri: str, dir_type: Path | str | None = None) -> Path:
    """
    Resolve any of:
        - absolute filesystem path  → returned as-is
        - `output://` URI         → resolved under OUTPUT_ROOT
        - relative path             → resolved under OUTPUT_ROOT (traversal-safe)
    """
    if not isinstance(path_or_uri, str) or not path_or_uri.strip():
        raise ValueError("resolve_local_path expects a non-empty string")

    s = path_or_uri.strip()
    p = Path(s)

    if p.is_absolute():
        return p

    if s.startswith(f"{OUTPUT_SCHEME}://"):
        out = from_output_uri(s, dir_type=dir_type)
        logger.debug(f"[storage][resolve_local_path] uri={s} -> {out}")
        return out

    out = _resolve_under_output_root(s, dir_type=dir_type)
    logger.debug(f"[storage][resolve_local_path] rel={s} -> {out}")
    return out


# ─── Per-document deterministic directory API ───

def doc_dir(doc_id: int, dir_type: Path | str | None = None) -> Path:
    """Return (and create) `<OUTPUT_ROOT>/documents/<doc_id:09d>/`."""
    out = mkdir_and_return_documents_path(dir_type) / f"{int(doc_id):09d}"
    out.mkdir(parents=True, exist_ok=True)
    return out


def sourcefile_dir(doc_id: int, dir_type: Path | str | None = None) -> Path:
    """Return (and create) `<OUTPUT_ROOT>/documents/<doc_id:09d>/SOURCEFILE/`."""
    out = doc_dir(doc_id, dir_type=dir_type) / SOURCEFILE_DIRNAME
    out.mkdir(parents=True, exist_ok=True)
    return out


def page_dir(doc_id: int, page_number: int, dir_type: Path | str | None = None) -> Path:
    """
    Return (and create) the page directory: `…/<doc_id:09d>/<page_number:04d>/`

    Args:
        page_number: 1-based page index (raises ValueError if < 1).
    """
    if int(page_number) <= 0:
        raise ValueError(f"page_number must be >= 1; got {page_number}")
    out = doc_dir(doc_id, dir_type=dir_type) / f"{int(page_number):04d}"
    out.mkdir(parents=True, exist_ok=True)
    return out


def page_layout_dir(doc_id: int, page_number: int, dir_type: Path | str | None = None) -> Path:
    """Return (and create) the per-page layout/diagnostics directory."""
    out = page_dir(doc_id, page_number, dir_type=dir_type) / LAYOUT_DIRNAME
    out.mkdir(parents=True, exist_ok=True)
    return out


def page_prefix(page_number: int) -> str:
    """Filename prefix for per-page artifacts, e.g. `p0001`."""
    if int(page_number) <= 0:
        raise ValueError(f"page_number must be >= 1; got {page_number}")
    return f"p{int(page_number):04d}"


def page_source_image_path(
    doc_id: int,
    page_number: int,
    *,
    ext: str = ".png",
    dir_type: Path | str | None = None,
) -> Path:
    """Deterministic path for the persisted page source image."""
    if not ext.startswith("."):
        ext = "." + ext
    return page_dir(doc_id, page_number, dir_type=dir_type) / f"{page_prefix(page_number)}_source{ext}"


# ─── DB-free helpers ───

def make_doc_key(original_filename: str, *, now=None) -> str:
    """
    Create a human-meaningful DB-free document folder key.

    Format: `YYYYMMDDTHHMMSS_<sanitized_filename>`
    Intended to replace PID-based folder naming for test runs.
    """
    from datetime import datetime

    ts   = (now or datetime.now()).strftime("%Y%m%dT%H%M%S")
    base = Path(original_filename).name or "document"
    safe = "".join(c if (c.isalnum() or c in "._-") else "_" for c in base)
    safe = safe.strip("._-") or "document"
    return f"{ts}_{safe}"


def test_doc_root(doc_key: str, *, dir_type: Path | str | None = None) -> Path:
    """
    Root folder for a DB-free document run, aligned to the DB-backed layout.

        <OUTPUT_TEMP>/documents/<doc_key>/SOURCEFILE/
    """
    root     = resolve_and_mkdir_output(dir_type or OUTPUT_TEMP)
    doc_root = root / "documents" / doc_key
    (doc_root / "SOURCEFILE").mkdir(parents=True, exist_ok=True)
    return doc_root


# ─── Rendering / splitting utilities ───

def split_pdf_or_image(
    raw_uri: str,
    *,
    dpi: int = 300,
    min_dpi: int = 150,
    max_dpi: int = 600,
    max_pixels: int = 25_000_000,
) -> tuple[list[np.ndarray], list[int]]:
    """
    Load a PDF/TIFF/image and return page images (BGR) plus per-page DPI.

    Accepted input for raw_uri:
        output:// URI, absolute path, or relative-to-output path.

    Notes:
        PDF rendering uses PyMuPDF if installed, falling back to pypdfium2.
        TIFF multi-frame decoding uses Pillow.
        Single images are loaded via OpenCV.
        For PDFs, DPI is *applied* at render time and may be reduced per page
        via a pixel-budget heuristic. For raster images, DPI is *inferred*
        (from metadata if available, otherwise an A4 long-edge heuristic) but
        pixels are not resampled.

    Args:
        raw_uri:    StorageURI or local path to a PDF, TIFF, or image.
        dpi:        Requested render DPI (default 300).
        min_dpi:    Lower bound for DPI after heuristics (default 150).
        max_dpi:    Upper bound for DPI after heuristics (default 600).
        max_pixels: Per-page pixel budget for PDF rendering. DPI is scaled down
                    for any page that would exceed this at the requested DPI.

    Returns:
        (images_bgr, dpis_used) — one entry per page/frame.

    Raises:
        FileNotFoundError: raw_uri cannot be resolved to an existing file.
        RuntimeError:      required dependency missing for PDF/TIFF.
        ValueError:        image cannot be decoded.
    """
    raw_path = resolve_local_path(raw_uri)

    if not raw_path.exists():
        logger.error(
            f"[S0][split_pdf_or_image] Missing input file. raw_uri={raw_uri} resolved={raw_path}"
        )
        raise FileNotFoundError(str(raw_path))

    raw_str = str(raw_path)
    lower = raw_str.lower()

    def _clamp_dpi(v: int) -> int:
        return int(max(min_dpi, min(max_dpi, int(v))))

    def _dpi_for_pdf_page(width_pt: float, height_pt: float) -> int:
        """
        Compute a safe DPI for a PDF page under the pixel budget.
        PDF coordinates are in points (1 pt = 1/72 inch).
        """
        w_in = max(0.01, float(width_pt) / 72.0)
        h_in = max(0.01, float(height_pt) / 72.0)
        requested = _clamp_dpi(dpi)
        if (requested * w_in) * (requested * h_in) <= max_pixels:
            return requested
        import math
        safe = int(math.floor(math.sqrt(max_pixels / (w_in * h_in))))
        return _clamp_dpi(safe)

    def _infer_raster_dpi(path: Path, arr_bgr: np.ndarray) -> int:
        """
        Infer DPI from image metadata; fall back to an A4 long-edge heuristic.
        Heuristic: assume the long side ≈ 11.69 inches (A4), yielding a stable
        estimate for scans when metadata is absent.
        """
        try:
            from PIL import Image
            with Image.open(path) as im:
                info    = getattr(im, "info", {}) or {}
                meta_dpi = info.get("dpi")
                if isinstance(meta_dpi, tuple) and len(meta_dpi) >= 2:
                    return _clamp_dpi(int(round(float(meta_dpi[0]) or dpi)))
                if isinstance(meta_dpi, (int, float)):
                    return _clamp_dpi(int(round(float(meta_dpi))))
        except Exception:
            pass
        h, w = arr_bgr.shape[:2]
        est  = int(round(float(max(w, h)) / 11.69))
        return _clamp_dpi(est or dpi)

    # ─── PDF: PyMuPDF first, then pypdfium2 ───
    # Prefer PyMuPDF (widely used, reliable, fast); fall back to pypdfium2 to
    # avoid a hard dependency on a single renderer during development.
    if lower.endswith(".pdf"):
        try:
            import pymupdf

            doc = pymupdf.open(raw_str)
            out_imgs: list[np.ndarray] = []
            out_dpis: list[int]        = []

            for i in range(doc.page_count):
                page  = doc.load_page(i)
                rect  = page.rect
                dpi_i = _dpi_for_pdf_page(rect.width, rect.height)
                zoom  = float(dpi_i) / 72.0
                pix   = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), alpha=False)
                arr   = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)

                if pix.n == 3:
                    out_imgs.append(cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))
                elif pix.n == 4:
                    out_imgs.append(cv2.cvtColor(arr[:, :, :3], cv2.COLOR_RGB2BGR))
                else:
                    raise RuntimeError(f"Unexpected PyMuPDF pixmap channels: {pix.n}")

                out_dpis.append(int(dpi_i))

            logger.info(
                f"[S0][split_pdf_or_image] Rendered PDF pages={doc.page_count} "
                f"requested_dpi={dpi} used_dpi_range={min(out_dpis)}..{max(out_dpis)}"
            )
            return out_imgs, out_dpis

        except ModuleNotFoundError:
            logger.warning(
                f"[S0][split_pdf_or_image] PyMuPDF not found; falling back to pypdfium2 "
                f"raw_uri={raw_uri}"
            )
            try:
                from pypdfium2 import PdfDocument
            except Exception as e:
                raise RuntimeError(
                    "PDF splitting requires a renderer. Install one of:\n"
                    "  - pymupdf  (recommended): pip install pymupdf\n"
                    "  - pypdfium2: pip install pypdfium2"
                ) from e

            pdf = PdfDocument(raw_str)
            out_imgs = []
            out_dpis = []

            for page in pdf:
                w_pt, h_pt = page.get_size()
                dpi_i = _dpi_for_pdf_page(w_pt, h_pt)
                scale = float(dpi_i) / 72.0
                arr_rgb = np.array(page.render(scale=scale).to_pil())
                out_imgs.append(cv2.cvtColor(arr_rgb, cv2.COLOR_RGB2BGR))
                out_dpis.append(int(dpi_i))

            logger.info(
                f"[S0][split_pdf_or_image] Rendered PDF (pypdfium2) pages={len(out_imgs)} "
                f"requested_dpi={dpi} used_dpi_range={min(out_dpis)}..{max(out_dpis)}"
            )
            return out_imgs, out_dpis

    # ─── TIFF (multi-frame) ───
    if lower.endswith((".tif", ".tiff")):
        try:
            from PIL import Image
        except Exception as e:
            raise RuntimeError("Pillow is required for TIFF splitting") from e

        im       = Image.open(raw_str)
        out_imgs = []
        out_dpis = []

        for i in range(getattr(im, "n_frames", 1)):
            im.seek(i)
            arr_bgr = cv2.cvtColor(np.array(im.convert("RGB")), cv2.COLOR_RGB2BGR)
            out_imgs.append(arr_bgr)
            out_dpis.append(_infer_raster_dpi(raw_path, arr_bgr))

        logger.info(
            f"[S0][split_pdf_or_image] Decoded TIFF frames={len(out_imgs)} "
            f"inferred_dpi_range={min(out_dpis)}..{max(out_dpis)}"
        )
        return out_imgs, out_dpis

    # ─── Single raster image ───
    arr = cv2.imread(raw_str)
    if arr is None:
        raise ValueError(f"Unreadable image at {raw_str}")

    dpi_i = _infer_raster_dpi(raw_path, arr)
    logger.info(
        f"[S0][split_pdf_or_image] Decoded image size={arr.shape[1]}x{arr.shape[0]} "
        f"inferred_dpi={dpi_i}"
    )
    return [arr], [dpi_i]


def write_page_images(
    doc_id: int,
    page_imgs_bgr: list[np.ndarray],
    *,
    ext: str = ".png",
    dir_type: Path | str | None = None,
) -> list[str]:
    """
    Persist page images deterministically under each page folder.

    Written to: `<OUTPUT_ROOT>/documents/<doc_id:09d>/<page_number:04d>/p<page_number>_source{ext}`

    Args:
        doc_id:         Document integer ID.
        page_imgs_bgr:  List of OpenCV-style BGR arrays.
        ext:            File extension (default `.png`).
        dir_type:       Output root override (for tests).

    Returns:
        List of `output://` URIs corresponding to written images.
    """
    if not page_imgs_bgr:
        return []

    uris: list[str] = []
    for i0, bgr in enumerate(page_imgs_bgr):
        page_number = i0 + 1
        path = page_source_image_path(doc_id, page_number, ext=ext, dir_type=dir_type)
        path.parent.mkdir(parents=True, exist_ok=True)

        if not cv2.imwrite(str(path), bgr):
            raise IOError(
                f"Failed to write page image doc_id={doc_id} "
                f"page_number={page_number} path={path}"
            )
        uris.append(to_output_uri(path, dir_type=dir_type))

    logger.info(
        f"[S0][write_page_images] Wrote page_images doc_id={doc_id} "
        f"pages={len(page_imgs_bgr)} ext={ext}"
    )
    return uris


# ─── Geometry helpers ───

def crop(img: np.ndarray, bbox: list[float]) -> np.ndarray:
    """
    Crop a region from a BGR image using a normalized bbox [x0, y0, x1, y1] ∈ [0,1].
    """
    h, w = img.shape[:2]
    x0, y0, x1, y1 = [int(b * s) for b, s in zip(bbox, (w, h, w, h))]
    return img[max(0, y0):min(h, y1), max(0, x0):min(w, x1)]


def norm_bbox(page, abs_bbox: list[int]) -> list[float]:
    """Normalize a pixel bbox [x0, y0, x1, y1] to [0,1] using page.width/height."""
    x0, y0, x1, y1 = abs_bbox
    return [x0 / page.width, y0 / page.height, x1 / page.width, y1 / page.height]


def denorm_bbox(page, norm_bbox_vals: list[float]) -> list[int]:
    """Denormalize a [0,1] bbox to pixels using page.width/height."""
    return [
        int(b * s)
        for b, s in zip(norm_bbox_vals, (page.width, page.height, page.width, page.height))
    ]


def iou(b1: list[float], b2: list[float]) -> float:
    """Compute IoU between two normalized bboxes [x0, y0, x1, y1]."""
    xa, ya = max(b1[0], b2[0]), max(b1[1], b2[1])
    xb, yb = min(b1[2], b2[2]), min(b1[3], b2[3])
    inter  = max(0, xb - xa) * max(0, yb - ya)
    a1     = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2     = (b2[2] - b2[0]) * (b2[3] - b2[1])
    union  = a1 + a2 - inter
    return inter / union if union else 0.0


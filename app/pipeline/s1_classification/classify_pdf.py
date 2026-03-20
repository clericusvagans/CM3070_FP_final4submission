# app/pipeline/s1_classification/classify_pdf.py

"""
PDF page and document classification (native vs scanned vs OCR-layered).

Classifies PDF documents at page level and document level into a small
normalized vocabulary for downstream pipeline decisions.

Page.sourcetype (PDF inputs only)
----------------------------------
    "native-pdf"  Born-digital; selectable text is the primary content.
    "ocr-pdf"     Scanned raster background + a selectable OCR text layer.
    "image-pdf"   Scanned image-only; no selectable text layer.

Document.sourcetype (PDF inputs only)
--------------------------------------
    Same as page type when all pages agree; "mixed-pdf" when they differ.

Image coverage estimation
-------------------------
Coverage is estimated from the displayed bounding boxes of image objects,
not from intrinsic image dimensions. Intrinsic dimensions are misleading (e.g.
a high-resolution logo displayed very small). Using on-page bboxes and
computing a true union area gives a significantly more accurate measure of
visible raster content.

Backend adapters
----------------
Two adapters implement the same PdfInspector protocol:

    PdfInspectorPyMuPDF   Uses pymupdf (always available as a project dependency).
    PdfInspectorPDFium    Uses pypdfium2 (optional; lazy import).

Backend selection policy (get_inspector):
    "auto"    Prefer PDFium if installed; otherwise fall back to PyMuPDF.
    "pymupdf" Always PyMuPDF.
    "pdfium"  Always PDFium (raises RuntimeError if pypdfium2 not installed).

DB contract
-----------
Per-page:    Page.sourcetype
Per-document: Document.sourcetype
Diagnostics: doc.doc_metadata["pdf_classification"] = {"backend": …, "issues": […]}
"""

from collections import Counter
from contextlib import AbstractContextManager, contextmanager
from pathlib import Path
from typing import Any, Iterable, Iterator, Literal, Protocol, TypeVar

import pymupdf
from sqlalchemy.orm import Session

from app.models import Document, Page
from app.utils.logger import logger
from app.utils.storage import resolve_local_path


PdfPageType = Literal["native-pdf", "ocr-pdf", "image-pdf"]
PdfDocType  = Literal["native-pdf", "ocr-pdf", "image-pdf", "mixed-pdf"]
PdfBackend  = Literal["auto", "pymupdf", "pdfium"]

TDoc  = TypeVar("TDoc")
TPage = TypeVar("TPage")


# ─── Diagnostics helper ───

def _add_issue(issues: list[dict[str, Any]], code: str, **fields: Any) -> None:
    """Append a compact structured diagnostic entry to the issues list."""
    entry: dict[str, Any] = {"code": code}
    entry.update(fields)
    issues.append(entry)


# ─── Inspector protocol ───

class PdfInspector(Protocol[TDoc, TPage]):
    """
    Minimal backend-agnostic adapter interface for PDF introspection.

    Isolates backend specifics (PyMuPDF vs PDFium) so that classification and
    orchestration logic remains backend-independent and inspectors can be
    swapped with fakes/mocks in tests.

    Implementations must provide:
        open_document(bytes)         Context manager yielding a document handle.
        page_count(doc)              Total page count.
        get_page(doc, index)         Page handle by index.
        has_text(page)               True if selectable non-whitespace text exists.
        displayed_image_ratio(page)  Union(image_bboxes) / page_area in [0, 1].
    """
    name: str

    def available(self) -> bool: ...
    def open_document(self, pdf_bytes: bytes) -> AbstractContextManager[TDoc]: ...
    def page_count(self, doc: TDoc) -> int: ...
    def get_page(self, doc: TDoc, index: int) -> TPage: ...
    def has_text(self, page: TPage) -> bool: ...
    def displayed_image_ratio(self, page: TPage) -> float: ...


# ─── PyMuPDF inspector ───

class PdfInspectorPyMuPDF(PdfInspector[pymupdf.Document, pymupdf.Page]):
    """
    PyMuPDF implementation of PdfInspector.

    Text detection:   page.get_text("text") — cheap presence signal.
    Image coverage:   page.get_text("dict") blocks where type==1 (image);
                      union area of displayed bboxes / page_area.
    Encryption:       doc.needs_pass + authenticate("").
    """

    name = "pymupdf"

    def available(self) -> bool:
        return True   # pymupdf is a project dependency; import already succeeded

    @contextmanager
    def open_document(self, pdf_bytes: bytes) -> Iterator[pymupdf.Document]:
        doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
        try:
            yield doc
        finally:
            doc.close()

    def page_count(self, doc: pymupdf.Document) -> int:
        return int(doc.page_count)

    def get_page(self, doc: pymupdf.Document, index: int) -> pymupdf.Page:
        return doc.load_page(index)

    def has_text(self, page: pymupdf.Page) -> bool:
        try:
            return bool((page.get_text("text") or "").strip())
        except Exception:
            return False

    def displayed_image_ratio(self, page: pymupdf.Page) -> float:
        """
        Estimate fraction of the page occupied by displayed images.
        Uses displayed bboxes (not intrinsic image dimensions) to avoid
        overcounting logos and other small decorative images.
        """
        try:
            rect      = page.rect
            page_area = float(rect.width * rect.height)
            if page_area <= 0:
                return 0.0

            rects: list[tuple[float, float, float, float]] = []
            for b in page.get_text("dict").get("blocks", []):
                if b.get("type") != 1:   # 1 = image block
                    continue
                bbox = b.get("bbox")
                if not bbox or len(bbox) != 4:
                    continue
                x0, y0, x1, y1 = map(float, bbox)
                if ((x1 - x0) * (y1 - y0)) / page_area <= 0.002:
                    continue   # skip negligible images
                rects.append((x0, y0, x1, y1))

            if not rects:
                return 0.0
            return min(max(_rect_union_area(rects) / page_area, 0.0), 1.0)
        except Exception:
            return 0.0


# ─── PDFium inspector ───

class PdfInspectorPDFium(PdfInspector[Any, Any]):
    """
    PDFium implementation via the optional `pypdfium2` dependency (lazy import).

    Text detection:    page.get_textpage().get_text_range().strip()
    Image coverage:    enumerate page objects of type "image"; union bbox area.
    Encryption:        PDFium encryption APIs vary across versions; failures are
                       surfaced to the caller for graceful fallback.
    """

    name = "pdfium"

    def __init__(self) -> None:
        try:
            import pypdfium2 as pdfium  # type: ignore
        except Exception:
            pdfium = None
        self._pdfium = pdfium

    def available(self) -> bool:
        return self._pdfium is not None

    @contextmanager
    def open_document(self, pdf_bytes: bytes) -> Iterator[Any]:
        if self._pdfium is None:
            raise RuntimeError("pypdfium2 is not installed.")
        pdf = self._pdfium.PdfDocument(pdf_bytes)
        try:
            yield pdf
        finally:
            try:
                pdf.close()
            except Exception:
                pass

    def page_count(self, doc: Any) -> int:
        return int(len(doc))

    def get_page(self, doc: Any, index: int) -> Any:
        return doc[index]

    def has_text(self, page: Any) -> bool:
        try:
            return bool((page.get_textpage().get_text_range() or "").strip())
        except Exception:
            return False

    def displayed_image_ratio(self, page: Any) -> float:
        """Estimate image coverage using PDFium page objects. Falls back to 0.0 on errors."""
        try:
            w, h      = page.get_size()
            page_area = float(w * h)
            if page_area <= 0:
                return 0.0

            rects: list[tuple[float, float, float, float]] = []
            for obj in page.get_objects():
                try:
                    if getattr(obj, "type", None) != "image":
                        continue
                    bbox = obj.get_bounds()   # (l, b, r, t) in PDFium coords
                    if not bbox or len(bbox) != 4:
                        continue
                    x0, y0, x1, y1 = _convert_pdfium_pos_to_topleft_xyxy(bbox, page_h=h)
                    if ((x1 - x0) * (y1 - y0)) / page_area <= 0.002:
                        continue
                    rects.append((x0, y0, x1, y1))
                except Exception:
                    continue

            if not rects:
                return 0.0
            return min(max(_rect_union_area(rects) / page_area, 0.0), 1.0)
        except Exception:
            return 0.0


# ─── Geometry helpers ───

def _merge_y_intervals(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Merge overlapping 1D y-intervals (used in the sweep-line union area calculation)."""
    merged: list[tuple[float, float]] = []
    for a, b in sorted(intervals):
        if not merged or a > merged[-1][1]:
            merged.append((a, b))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], b))
    return merged


def _rect_union_area(rects: Iterable[tuple[float, float, float, float]]) -> float:
    """
    Compute union area of axis-aligned rectangles using a sweep line on x.
    Each rect is (x0, y0, x1, y1).
    """
    rect_list = [
        (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))
        for x0, y0, x1, y1 in rects
    ]
    if not rect_list:
        return 0.0

    xs    = sorted({x0 for x0, _, x1, _ in rect_list} | {x1 for _, _, x1, _ in rect_list})
    area  = 0.0
    for i in range(len(xs) - 1):
        x_left, x_right = xs[i], xs[i + 1]
        if x_right <= x_left:
            continue
        y_intervals = [
            (y0, y1)
            for x0, y0, x1, y1 in rect_list
            if x0 <= x_left and x1 >= x_right
        ]
        if not y_intervals:
            continue
        y_total = sum(max(0.0, b - a) for a, b in _merge_y_intervals(y_intervals))
        area   += (x_right - x_left) * y_total
    return area


def _convert_pdfium_pos_to_topleft_xyxy(
    bbox_lbrt: tuple[float, float, float, float],
    *,
    page_h: float,
) -> tuple[float, float, float, float]:
    """
    Convert PDFium (left, bottom, right, top) to top-left-origin (x0, y0, x1, y1).
    PDFium y-axis grows upward; this flips it so y grows downward.
    """
    l, b, r, t = bbox_lbrt
    x0, x1 = float(l), float(r)
    y0 = float(page_h - t)
    y1 = float(page_h - b)
    return (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))


# ─── Backend selection ───

def get_inspector(backend: PdfBackend) -> PdfInspector[Any, Any]:
    """
    Return a PDF inspector for the requested backend.

    "auto"    Prefer PDFium if installed (often better for geometry/object
              inspection); fall back to PyMuPDF (guaranteed dependency).
    "pymupdf" Always PyMuPDF.
    "pdfium"  Always PDFium; raises RuntimeError if pypdfium2 not installed.
    """
    pym    = PdfInspectorPyMuPDF()
    pdfium = PdfInspectorPDFium()

    if backend == "pymupdf":
        return pym
    if backend == "pdfium":
        if not pdfium.available():
            raise RuntimeError("Requested backend 'pdfium' but pypdfium2 is not installed.")
        return pdfium
    return pdfium if pdfium.available() else pym


# ─── Classification rules ───

def _classify_pdf_page_backend(*, has_text: bool, image_ratio: float) -> PdfPageType:
    """
    Backend-agnostic page classification decision tree.

    has_text + low raster coverage (<5%)  → "native-pdf"
    has_text + high raster coverage       → "ocr-pdf"
    no selectable text                    → "image-pdf"
    """
    if has_text:
        return "native-pdf" if image_ratio < 0.05 else "ocr-pdf"
    return "image-pdf"


def _classify_pdf_document(counts: Counter[str], total_pages: int) -> PdfDocType:
    """Aggregate per-page classifications: unanimous type → that type; mixed → "mixed-pdf"."""
    if total_pages <= 0:
        return "image-pdf"
    for t in ("native-pdf", "ocr-pdf", "image-pdf"):
        if counts.get(t, 0) == total_pages:
            return t  # type: ignore[return-value]
    return "mixed-pdf"


# ─── Pipeline entrypoint ───

def classify_pdf(
    db: Session,
    doc: Document,
    pages: list[Page],
    pdf_path: str | Path,
    *,
    backend: PdfBackend = "auto",
) -> None:
    """
    Pipeline stage S1: classify PDF pages and persist sourcetype fields.

    Updates:
        Page.sourcetype          Per-page classification.
        Document.sourcetype      Document-level aggregate.
        doc.doc_metadata["pdf_classification"]  Minimal diagnostics.

    Failure handling:
        Encrypted / unreadable PDF  → all pages and doc fall back to "image-pdf".
        len(pages) != PDF page count → classify overlapping pages; extras fall
                                       back to "image-pdf". Diagnostic recorded.
        Unexpected backend error     → all pages/doc fall back to "image-pdf".
    """
    pdf_path = resolve_local_path(pdf_path)
    issues: list[dict[str, Any]] = []

    try:
        pdf_bytes = pdf_path.read_bytes()
    except (FileNotFoundError, PermissionError) as e:
        raise RuntimeError(f"Cannot read PDF file: {e}")

    inspector = get_inspector(backend)
    page_labels: list[PdfPageType] = []

    try:
        with inspector.open_document(pdf_bytes) as pdf:
            # Encrypted PDF handling (PyMuPDF only — PDFium APIs vary by version).
            if isinstance(inspector, PdfInspectorPyMuPDF):
                try:
                    if pdf.needs_pass and not pdf.authenticate(""):
                        _add_issue(issues, "encrypted_or_unreadable", file=str(pdf_path))
                        for p in pages:
                            p.sourcetype = "image-pdf"
                        doc.sourcetype   = "image-pdf"
                        doc.doc_metadata = {"pdf_classification": {"backend": inspector.name, "issues": issues}}
                        db.flush()
                        return
                except Exception:
                    pass   # check failed; proceed and let normal error handling apply

            total = inspector.page_count(pdf)
            if total <= 0:
                raise RuntimeError("PDF has 0 pages")

            n = min(len(pages), total)
            if len(pages) != total:
                _add_issue(
                    issues, "page_count_mismatch",
                    db_pages=len(pages), pdf_pages=total, classified_pages=n,
                )

            for i in range(n):
                page_handle = inspector.get_page(pdf, i)
                label       = _classify_pdf_page_backend(
                    has_text    =inspector.has_text(page_handle),
                    image_ratio =inspector.displayed_image_ratio(page_handle),
                )
                page_labels.append(label)
                pages[i].sourcetype = label

            for j in range(n, len(pages)):   # extra DB pages beyond the PDF
                pages[j].sourcetype = "image-pdf"
                page_labels.append("image-pdf")

    except Exception as e:
        logger.exception(
            f"PDF classification failed for doc={getattr(doc, 'id', None)} "
            f"backend={inspector.name}: {type(e).__name__}: {e}"
        )
        _add_issue(issues, "backend_error", error_type=type(e).__name__, error=str(e))
        for p in pages:
            p.sourcetype = "image-pdf"
        doc.sourcetype   = "image-pdf"
        doc.doc_metadata = {"pdf_classification": {"backend": inspector.name, "issues": issues}}
        db.flush()
        return

    doc.sourcetype   = _classify_pdf_document(Counter(page_labels), len(page_labels))
    doc.doc_metadata = {"pdf_classification": {"backend": inspector.name, "issues": issues}}

    logger.info(
        f"[Orchestrator][S1] PDF classified doc={getattr(doc, 'id', None)} "
        f"as {doc.sourcetype} (pages={len(page_labels)}) "
        f"[backend={inspector.name}]"
    )
    db.flush()


def classify_pdf_bytes(
    pdf_bytes: bytes,
    *,
    backend: PdfBackend = "auto",
) -> tuple[str, list[str], str]:
    """
    Classify a PDF without DB/ORM objects. Useful for testing.

    Returns:
        (doc_sourcetype, page_sourcetypes, backend_name)

    Falls back to ("image-pdf", [], backend_name) on encrypted or unreadable PDFs
    and on unexpected backend errors.
    """
    inspector = get_inspector(backend)
    page_labels: list[PdfPageType] = []

    try:
        with inspector.open_document(pdf_bytes) as pdf:
            if isinstance(inspector, PdfInspectorPyMuPDF):
                try:
                    if pdf.needs_pass and not pdf.authenticate(""):
                        return "image-pdf", [], inspector.name
                except Exception:
                    pass

            total = inspector.page_count(pdf)
            if total <= 0:
                return "image-pdf", [], inspector.name

            for i in range(total):
                page_handle = inspector.get_page(pdf, i)
                label = _classify_pdf_page_backend(
                    has_text    =inspector.has_text(page_handle),
                    image_ratio =inspector.displayed_image_ratio(page_handle),
                )
                page_labels.append(label)

    except Exception:
        return "image-pdf", [], inspector.name

    doc_type = _classify_pdf_document(Counter(page_labels), len(page_labels))
    return doc_type, [str(x) for x in page_labels], inspector.name



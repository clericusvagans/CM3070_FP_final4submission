# tests/test_storage.py
"""
Unit tests for app/utils/storage.py.

Covers:
    URI round-trip         to_output_uri → from_output_uri → resolve_local_path
    Two-slash URI form     output://documents/...  (urlparse puts first segment in netloc)
    Traversal guard        _resolve_under_output_root raises on path-escape attempts
    Directory layout       doc_dir, sourcefile_dir, page_dir, page_layout_dir
    Naming conventions     page_prefix, page_source_image_path
    make_doc_key           deterministic timestamp+filename sanitisation
    Geometry helpers       iou, norm_bbox, denorm_bbox, crop

All tests are filesystem-only and use pytest's tmp_path fixture so they
never touch the real OUTPUT_ROOT.  No DB, no OpenCV GPU, no Paddle required.

Run with:
    python -m pytest tests/test_storage.py
    python -m pytest tests/test_storage.py -v
"""

import re
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from app.utils.storage import (
    SOURCEFILE_DIRNAME,
    LAYOUT_DIRNAME,
    SUPPORTED_EXTS,
    doc_dir,
    from_output_uri,
    iou,
    make_doc_key,
    norm_bbox,
    denorm_bbox,
    page_dir,
    page_layout_dir,
    page_prefix,
    page_source_image_path,
    resolve_local_path,
    sourcefile_dir,
    to_output_uri,
)


# ─── URI round-trip ───

class TestUriRoundTrip:
    """
    to_output_uri and from_output_uri must be exact inverses.

    Two URI forms exist:
        Triple-slash  output:///documents/000000001/SOURCEFILE/file.pdf
            urlparse → scheme='output', netloc='', path='documents/...'
        Two-slash     output://documents/000000001/SOURCEFILE/file.pdf
            urlparse → scheme='output', netloc='documents', path='/000000001/...'

    from_output_uri handles both; to_output_uri always produces triple-slash.
    """

    def test_triple_slash_round_trip(self, tmp_path):
        target = tmp_path / "documents" / "000000001" / "SOURCEFILE" / "file.pdf"
        target.parent.mkdir(parents=True)
        target.touch()

        uri    = to_output_uri(target, dir_type=tmp_path)
        result = from_output_uri(uri, dir_type=tmp_path)
        assert result == target.resolve()

    def test_uri_has_output_scheme(self, tmp_path):
        target = tmp_path / "documents" / "000000001" / "file.txt"
        target.parent.mkdir(parents=True)
        target.touch()

        uri = to_output_uri(target, dir_type=tmp_path)
        assert uri.startswith("output://")

    def test_triple_slash_canonical_form(self, tmp_path):
        """to_output_uri must always produce triple-slash form."""
        target = tmp_path / "documents" / "000000001" / "file.txt"
        target.parent.mkdir(parents=True)
        target.touch()

        uri = to_output_uri(target, dir_type=tmp_path)
        assert uri.startswith("output:///"), f"Expected triple-slash, got: {uri}"

    def test_two_slash_form_resolved_correctly(self, tmp_path):
        """
        from_output_uri must handle the two-slash form where urlparse puts
        the first path segment into netloc.
        """
        target = tmp_path / "documents" / "000000042" / "SOURCEFILE" / "doc.pdf"
        target.parent.mkdir(parents=True)
        target.touch()

        two_slash_uri = "output://documents/000000042/SOURCEFILE/doc.pdf"
        result = from_output_uri(two_slash_uri, dir_type=tmp_path)
        assert result == target.resolve()

    def test_resolve_local_path_accepts_uri(self, tmp_path):
        target = tmp_path / "documents" / "000000001" / "page.png"
        target.parent.mkdir(parents=True)
        target.touch()

        uri    = to_output_uri(target, dir_type=tmp_path)
        result = resolve_local_path(uri, dir_type=tmp_path)
        assert result == target.resolve()

    def test_resolve_local_path_accepts_absolute_path(self, tmp_path):
        target = tmp_path / "somefile.txt"
        target.touch()
        assert resolve_local_path(str(target)) == target

    def test_resolve_local_path_accepts_relative_path(self, tmp_path):
        target = tmp_path / "subdir" / "file.txt"
        target.parent.mkdir()
        target.touch()

        result = resolve_local_path("subdir/file.txt", dir_type=tmp_path)
        assert result == target.resolve()

    def test_wrong_scheme_raises(self, tmp_path):
        with pytest.raises(ValueError, match="Not an output://"):
            from_output_uri("http://example.com/file.pdf", dir_type=tmp_path)

    def test_empty_string_raises(self, tmp_path):
        with pytest.raises(ValueError):
            resolve_local_path("", dir_type=tmp_path)

    def test_malformed_uri_raises(self, tmp_path):
        with pytest.raises(ValueError, match="Malformed"):
            from_output_uri("output://", dir_type=tmp_path)


# ─── Traversal guard ───

class TestTraversalGuard:
    """
    Paths that escape OUTPUT_ROOT via ../.. must be rejected.
    This protects against URI-derived paths pointing outside the output tree.
    """

    def test_path_traversal_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="escapes OUTPUT_ROOT"):
            resolve_local_path("../../etc/passwd", dir_type=tmp_path)

    def test_path_inside_root_accepted(self, tmp_path):
        target = tmp_path / "documents" / "safe.txt"
        target.parent.mkdir(parents=True)
        target.touch()
        result = resolve_local_path("documents/safe.txt", dir_type=tmp_path)
        assert result == target.resolve()


# ─── Directory layout ───

class TestDirectoryLayout:
    """
    Verify the deterministic on-disk layout used by all pipeline stages.

    Convention:
        <root>/documents/<doc_id:09d>/
            SOURCEFILE/
            <page_number:04d>/
                LAYOUT/
    """

    def test_doc_dir_nine_digit_zero_padded(self, tmp_path):
        path = doc_dir(1, dir_type=tmp_path)
        assert path.name == "000000001"
        assert path.exists()

    def test_doc_dir_large_id(self, tmp_path):
        path = doc_dir(123456789, dir_type=tmp_path)
        assert path.name == "123456789"

    def test_doc_dir_under_documents(self, tmp_path):
        path = doc_dir(1, dir_type=tmp_path)
        assert path.parent.name == "documents"

    def test_sourcefile_dir_name(self, tmp_path):
        path = sourcefile_dir(1, dir_type=tmp_path)
        assert path.name == SOURCEFILE_DIRNAME
        assert path.parent == doc_dir(1, dir_type=tmp_path)

    def test_page_dir_four_digit_zero_padded(self, tmp_path):
        path = page_dir(1, 3, dir_type=tmp_path)
        assert path.name == "0003"

    def test_page_dir_zero_raises(self, tmp_path):
        """page_number is 1-based; 0 must be rejected."""
        with pytest.raises(ValueError, match=">= 1"):
            page_dir(1, 0, dir_type=tmp_path)

    def test_page_dir_negative_raises(self, tmp_path):
        with pytest.raises(ValueError):
            page_dir(1, -1, dir_type=tmp_path)

    def test_page_layout_dir_under_page_dir(self, tmp_path):
        path = page_layout_dir(1, 2, dir_type=tmp_path)
        assert path.name == LAYOUT_DIRNAME
        assert path.parent == page_dir(1, 2, dir_type=tmp_path)

    def test_directories_created_on_call(self, tmp_path):
        path = page_dir(42, 7, dir_type=tmp_path)
        assert path.is_dir()


# ─── Naming conventions ───

class TestNamingConventions:

    @pytest.mark.parametrize("page_number,expected", [
        (1,    "p0001"),
        (9,    "p0009"),
        (10,   "p0010"),
        (999,  "p0999"),
        (1000, "p1000"),
    ])
    def test_page_prefix_format(self, page_number, expected):
        assert page_prefix(page_number) == expected

    def test_page_prefix_zero_raises(self):
        with pytest.raises(ValueError):
            page_prefix(0)

    def test_page_source_image_path_default_ext(self, tmp_path):
        path = page_source_image_path(1, 1, dir_type=tmp_path)
        assert path.name == "p0001_source.png"

    def test_page_source_image_path_custom_ext(self, tmp_path):
        path = page_source_image_path(1, 1, ext=".jpg", dir_type=tmp_path)
        assert path.name == "p0001_source.jpg"

    def test_page_source_image_path_ext_without_dot(self, tmp_path):
        """Passing ext without leading dot must still produce a valid filename."""
        path = page_source_image_path(1, 1, ext="tiff", dir_type=tmp_path)
        assert path.name == "p0001_source.tiff"

    def test_supported_extensions_set(self):
        assert ".pdf"  in SUPPORTED_EXTS
        assert ".png"  in SUPPORTED_EXTS
        assert ".tiff" in SUPPORTED_EXTS
        assert ".docx" not in SUPPORTED_EXTS


# ─── make_doc_key ───

class TestMakeDocKey:
    """
    make_doc_key produces a timestamp-prefixed, filesystem-safe folder name
    for DB-free test runs.
    """

    def test_format_timestamp_prefix(self):
        fixed = datetime(2024, 6, 15, 10, 30, 45)
        key   = make_doc_key("report.pdf", now=fixed)
        assert key.startswith("20240615T103045_")

    def test_filename_sanitised(self):
        fixed = datetime(2024, 1, 1, 0, 0, 0)
        key   = make_doc_key("my document (final).pdf", now=fixed)
        # Spaces and parentheses replaced with underscores
        assert " " not in key
        assert "(" not in key
        assert ")" not in key

    def test_alphanumeric_filename_unchanged(self):
        fixed = datetime(2024, 1, 1, 0, 0, 0)
        key   = make_doc_key("report2024.pdf", now=fixed)
        assert "report2024.pdf" in key

    def test_empty_filename_fallback(self):
        fixed = datetime(2024, 1, 1, 0, 0, 0)
        key   = make_doc_key("", now=fixed)
        assert "document" in key

    def test_key_contains_no_path_separators(self):
        fixed = datetime(2024, 1, 1, 0, 0, 0)
        key   = make_doc_key("path/to/file.pdf", now=fixed)
        assert "/" not in key.split("_", 1)[1]


# ─── Geometry helpers ───

class TestIoU:
    """
    IoU (intersection-over-union) for normalised bboxes [x0, y0, x1, y1].
    Values are in [0, 1].
    """

    def test_identical_boxes_is_one(self):
        b = [0.1, 0.1, 0.5, 0.5]
        assert iou(b, b) == pytest.approx(1.0)

    def test_non_overlapping_boxes_is_zero(self):
        b1 = [0.0, 0.0, 0.4, 0.4]
        b2 = [0.6, 0.6, 1.0, 1.0]
        assert iou(b1, b2) == pytest.approx(0.0)

    def test_partial_overlap(self):
        b1 = [0.0, 0.0, 0.5, 0.5]
        b2 = [0.25, 0.25, 0.75, 0.75]
        result = iou(b1, b2)
        assert 0.0 < result < 1.0

    def test_one_box_inside_other(self):
        outer = [0.0, 0.0, 1.0, 1.0]
        inner = [0.25, 0.25, 0.75, 0.75]
        result = iou(inner, outer)
        # inner area = 0.25, outer area = 1.0, union = 1.0 → IoU = 0.25
        assert result == pytest.approx(0.25)

    def test_result_in_unit_interval(self):
        import random
        rng = random.Random(0)
        for _ in range(50):
            coords = sorted(rng.random() for _ in range(4))
            b1 = [coords[0], coords[1], coords[2], coords[3]]
            b2 = [rng.random() * 0.5, rng.random() * 0.5,
                  0.5 + rng.random() * 0.5, 0.5 + rng.random() * 0.5]
            assert 0.0 <= iou(b1, b2) <= 1.0


class TestNormDenormBbox:
    """norm_bbox and denorm_bbox must be exact inverses at integer pixel coordinates."""

    @pytest.fixture
    def page(self):
        return SimpleNamespace(width=2481, height=3508)   # A4 at 300 DPI

    def test_round_trip(self, page):
        abs_bbox = [100, 200, 400, 600]
        norm     = norm_bbox(page, abs_bbox)
        result   = denorm_bbox(page, norm)
        assert result == abs_bbox

    def test_full_page_bbox(self, page):
        norm   = norm_bbox(page, [0, 0, page.width, page.height])
        result = denorm_bbox(page, norm)
        assert result == [0, 0, page.width, page.height]

    def test_normalised_values_in_unit_interval(self, page):
        norm = norm_bbox(page, [100, 200, 400, 600])
        assert all(0.0 <= v <= 1.0 for v in norm)


class TestCrop:
    """crop() must extract the correct sub-array from a BGR image."""

    def test_crop_top_left_quarter(self):
        from app.utils.storage import crop
        img    = np.zeros((100, 100, 3), dtype=np.uint8)
        img[:50, :50] = 255   # top-left quarter is white
        result = crop(img, [0.0, 0.0, 0.5, 0.5])
        assert result.shape == (50, 50, 3)
        assert result.min() == 255

    def test_crop_clamped_to_image_bounds(self):
        from app.utils.storage import crop
        img    = np.zeros((100, 100, 3), dtype=np.uint8)
        # bbox that would exceed image bounds — must be clamped
        result = crop(img, [0.8, 0.8, 1.5, 1.5])
        assert result.shape[0] <= 100
        assert result.shape[1] <= 100


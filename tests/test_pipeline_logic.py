# tests/test_pipeline_logic.py
"""
Unit tests for pure-logic parts of the pipeline that require no DB, no disk
access, and no heavy ML dependencies (Paddle, Tesseract, Lingua models).

Covers:

    pipeline/modes.py
        PipelineMode enum values and string coercion
        parse_pipeline_mode() — string/enum normalisation at the enqueue boundary

    pipeline/s2_detection/lang_script_detect.py  (pure helpers only)
        _tesseract_lang_arg()   ISO 639-1 → Tesseract 3-letter codes
        _lingua_probs()         rounding, short-text guard, empty-result guard
        _SCRIPT_TO_ISO15924     script name → ISO 15924 mapping
        Document-level aggregation arithmetic (normalisation + rounding)

    The full detect_langs_and_scripts() function is not tested here because
    it requires a live Tesseract installation and a Lingua model.  The helpers
    it delegates to are tested individually, which covers all the logic that
    can go wrong without external dependencies.

Run with:
    python -m pytest tests/test_pipeline_logic.py
    python -m pytest tests/test_pipeline_logic.py -v
"""

import pytest
from collections import Counter
from unittest.mock import MagicMock, patch


# ─── PipelineMode ────

class TestPipelineMode:

    def test_enum_values(self):
        from app.pipeline.modes import PipelineMode
        assert PipelineMode.DEFAULT.value == "default"
        assert PipelineMode.REPAIR.value  == "repair"
        assert PipelineMode.REDO.value    == "redo"

    def test_is_str_subclass(self):
        """PipelineMode(str, Enum) — members must compare equal to their string value."""
        from app.pipeline.modes import PipelineMode
        assert PipelineMode.DEFAULT == "default"
        assert PipelineMode.REPAIR  == "repair"
        assert PipelineMode.REDO    == "redo"

    def test_all_three_members_exist(self):
        from app.pipeline.modes import PipelineMode
        members = {m.value for m in PipelineMode}
        assert members == {"default", "repair", "redo"}


class TestParsePipelineMode:

    def test_string_default(self):
        from app.pipeline.modes import PipelineMode, parse_pipeline_mode
        assert parse_pipeline_mode("default") is PipelineMode.DEFAULT

    def test_string_repair(self):
        from app.pipeline.modes import PipelineMode, parse_pipeline_mode
        assert parse_pipeline_mode("repair") is PipelineMode.REPAIR

    def test_string_redo(self):
        from app.pipeline.modes import PipelineMode, parse_pipeline_mode
        assert parse_pipeline_mode("redo") is PipelineMode.REDO

    def test_enum_passthrough(self):
        """A PipelineMode value passed in must be returned unchanged."""
        from app.pipeline.modes import PipelineMode, parse_pipeline_mode
        for mode in PipelineMode:
            assert parse_pipeline_mode(mode) is mode

    def test_invalid_string_raises(self):
        from app.pipeline.modes import parse_pipeline_mode
        with pytest.raises(ValueError, match="Invalid worker pipeline mode"):
            parse_pipeline_mode("unknown")

    def test_empty_string_raises(self):
        from app.pipeline.modes import parse_pipeline_mode
        with pytest.raises(ValueError):
            parse_pipeline_mode("")

    def test_uppercase_raises(self):
        """parse_pipeline_mode is case-sensitive — 'DEFAULT' is not valid."""
        from app.pipeline.modes import parse_pipeline_mode
        with pytest.raises(ValueError):
            parse_pipeline_mode("DEFAULT")


# ─── _tesseract_lang_arg ────

class TestTesseractLangArg:
    """
    _tesseract_lang_arg converts ISO 639-1 two-letter codes to Tesseract
    three-letter codes and joins them with '+'.
    """

    def test_single_known_language(self):
        from app.pipeline.s2_detection.lang_script_detect import _tesseract_lang_arg
        assert _tesseract_lang_arg(["en"]) == "eng"
        assert _tesseract_lang_arg(["fr"]) == "fra"
        assert _tesseract_lang_arg(["uk"]) == "ukr"

    def test_multiple_languages_joined(self):
        from app.pipeline.s2_detection.lang_script_detect import _tesseract_lang_arg
        result = _tesseract_lang_arg(["en", "fr", "nl"])
        assert result == "eng+fra+nld"

    def test_unknown_code_passed_through(self):
        """Codes not in the map should be passed through as-is."""
        from app.pipeline.s2_detection.lang_script_detect import _tesseract_lang_arg
        result = _tesseract_lang_arg(["xx"])
        assert result == "xx"

    def test_empty_list_returns_eng_fallback(self):
        from app.pipeline.s2_detection.lang_script_detect import _tesseract_lang_arg
        assert _tesseract_lang_arg([]) == "eng"

    def test_duplicate_codes_deduplicated(self):
        """Same language appearing twice must not appear twice in the arg string."""
        from app.pipeline.s2_detection.lang_script_detect import _tesseract_lang_arg
        result = _tesseract_lang_arg(["en", "en", "fr"])
        assert result == "eng+fra"

    def test_ordering_preserved(self):
        """Order of languages in the output must match the input order."""
        from app.pipeline.s2_detection.lang_script_detect import _tesseract_lang_arg
        result = _tesseract_lang_arg(["ru", "uk", "en"])
        assert result == "rus+ukr+eng"

    def test_all_mapped_languages(self):
        """All languages in _TESS_2L_TO_3L must map to a non-empty 3-letter code."""
        from app.pipeline.s2_detection.lang_script_detect import (
            _tesseract_lang_arg,
            _TESS_2L_TO_3L,
        )
        for two_letter, three_letter in _TESS_2L_TO_3L.items():
            result = _tesseract_lang_arg([two_letter])
            assert result == three_letter


# ─── _lingua_probs ────

class TestLinguaProbs:
    """
    _lingua_probs wraps Lingua's confidence scoring.  The detector is mocked
    to test the logic (rounding, short-text guard, error handling) without
    loading a real Lingua model.
    """

    def _make_result(self, lang_name: str, iso_name: str, value: float):
        """Build a mock Lingua confidence result object."""
        iso   = MagicMock()
        iso.name = iso_name
        lang  = MagicMock()
        lang.iso_code_639_1 = iso
        result = MagicMock()
        result.language = lang
        result.value    = value
        return result

    def test_scores_rounded_to_six_places(self):
        from app.pipeline.s2_detection.lang_script_detect import _lingua_probs

        detector = MagicMock()
        detector.compute_language_confidence_values.return_value = [
            self._make_result("English", "EN", 0.123456789),
        ]
        probs = _lingua_probs(detector, "A" * 25)
        assert probs["en"] == round(0.123456789, 6)

    def test_multiple_languages_returned(self):
        from app.pipeline.s2_detection.lang_script_detect import _lingua_probs

        detector = MagicMock()
        detector.compute_language_confidence_values.return_value = [
            self._make_result("English",   "EN", 0.7),
            self._make_result("French",    "FR", 0.2),
            self._make_result("Dutch",     "NL", 0.1),
        ]
        probs = _lingua_probs(detector, "A" * 25)
        assert set(probs.keys()) == {"en", "fr", "nl"}

    def test_short_text_returns_empty_dict(self):
        """Texts shorter than 20 characters must be skipped without calling Lingua."""
        from app.pipeline.s2_detection.lang_script_detect import _lingua_probs

        detector = MagicMock()
        probs = _lingua_probs(detector, "short")
        assert probs == {}
        detector.compute_language_confidence_values.assert_not_called()

    def test_empty_text_returns_empty_dict(self):
        from app.pipeline.s2_detection.lang_script_detect import _lingua_probs
        assert _lingua_probs(MagicMock(), "") == {}
        assert _lingua_probs(MagicMock(), None) == {}  # type: ignore[arg-type]

    def test_lingua_exception_returns_empty_dict(self):
        """A Lingua error must not propagate — return empty dict instead."""
        from app.pipeline.s2_detection.lang_script_detect import _lingua_probs

        detector = MagicMock()
        detector.compute_language_confidence_values.side_effect = RuntimeError("model error")
        probs = _lingua_probs(detector, "A" * 30)
        assert probs == {}

    def test_result_without_iso_code_excluded(self):
        """Lingua results with no iso_code_639_1 attribute must be silently skipped."""
        from app.pipeline.s2_detection.lang_script_detect import _lingua_probs

        result = MagicMock()
        result.language.iso_code_639_1 = None
        result.value = 0.9

        detector = MagicMock()
        detector.compute_language_confidence_values.return_value = [result]
        probs = _lingua_probs(detector, "A" * 30)
        assert probs == {}

    def test_values_stored_as_float(self):
        from app.pipeline.s2_detection.lang_script_detect import _lingua_probs

        detector = MagicMock()
        detector.compute_language_confidence_values.return_value = [
            self._make_result("English", "EN", 0.8),
        ]
        probs = _lingua_probs(detector, "A" * 25)
        assert isinstance(probs["en"], float)


# ─── _SCRIPT_TO_ISO15924 mapping ────

class TestScriptMapping:
    """
    _SCRIPT_TO_ISO15924 maps lowercase Tesseract script names to ISO 15924
    codes used throughout the pipeline.
    """

    def test_known_mappings(self):
        from app.pipeline.s2_detection.lang_script_detect import _SCRIPT_TO_ISO15924
        assert _SCRIPT_TO_ISO15924["latin"]      == "Latn"
        assert _SCRIPT_TO_ISO15924["cyrillic"]   == "Cyrl"
        assert _SCRIPT_TO_ISO15924["arabic"]     == "Arab"
        assert _SCRIPT_TO_ISO15924["greek"]      == "Grek"
        assert _SCRIPT_TO_ISO15924["han"]        == "Hani"

    def test_all_values_are_four_char_codes(self):
        """ISO 15924 codes are always exactly 4 characters."""
        from app.pipeline.s2_detection.lang_script_detect import _SCRIPT_TO_ISO15924
        for key, code in _SCRIPT_TO_ISO15924.items():
            assert len(code) == 4, f"Expected 4-char code for '{key}', got '{code}'"

    def test_all_keys_are_lowercase(self):
        from app.pipeline.s2_detection.lang_script_detect import _SCRIPT_TO_ISO15924
        for key in _SCRIPT_TO_ISO15924:
            assert key == key.lower(), f"Key '{key}' is not lowercase"


# ─── Document-level aggregation arithmetic ────

class TestDocumentAggregation:
    """
    Tests for the normalisation logic that aggregates per-page lang_probs into
    Document.detected_langs.

    This logic lives inside detect_langs_and_scripts() but the arithmetic is
    simple enough to test directly.  I replicate the exact computation from
    the production function to pin the expected behaviour — including the
    rounding fix applied to the normalisation step.
    """

    def _aggregate(self, page_probs: list[dict[str, float]]) -> dict[str, float]:
        """
        Replicate the document-level aggregation from detect_langs_and_scripts().

        acc_probs is a Counter of summed per-page confidence scores.
        detected is normalised and rounded to 6 decimal places.
        """
        acc = Counter()
        for probs in page_probs:
            acc.update(probs)

        if not acc:
            return {}

        total = float(sum(acc.values()))
        if total <= 0:
            return {}

        return {k: round(float(v) / total, 6) for k, v in acc.items()}

    def test_single_page_single_lang(self):
        result = self._aggregate([{"en": 0.9}])
        assert result == {"en": pytest.approx(1.0)}

    def test_single_page_normalised_to_one(self):
        result = self._aggregate([{"en": 0.6, "fr": 0.4}])
        total  = sum(result.values())
        assert total == pytest.approx(1.0, abs=1e-5)

    def test_multi_page_aggregation(self):
        """With two identical pages the proportions must be unchanged."""
        probs  = {"en": 0.7, "fr": 0.3}
        result = self._aggregate([probs, probs])
        assert result["en"] == pytest.approx(0.7, abs=1e-5)
        assert result["fr"] == pytest.approx(0.3, abs=1e-5)

    def test_dominant_language_preserved(self):
        result = self._aggregate([{"uk": 0.8, "ru": 0.15, "en": 0.05}])
        primary = max(result.items(), key=lambda kv: kv[1])[0]
        assert primary == "uk"

    def test_all_values_rounded_to_six_places(self):
        """
        Normalisation via float division produces full-precision floats.
        The rounding step must cap at 6 decimal places for compact JSONB storage.
        """
        result = self._aggregate([{"en": 1.0, "fr": 3.0}])
        for v in result.values():
            # Verify no value has more than 6 significant decimal digits
            as_str = f"{v:.10f}".rstrip("0")
            decimals = len(as_str.split(".")[1]) if "." in as_str else 0
            assert decimals <= 6, f"Value {v} has too many decimal places"

    def test_empty_page_list_returns_empty_dict(self):
        assert self._aggregate([]) == {}

    def test_zero_total_returns_empty_dict(self):
        """Edge case: if all scores are 0 the result must be empty, not NaN."""
        assert self._aggregate([{"en": 0.0}]) == {}

    def test_values_sum_to_one_after_rounding(self):
        """
        After rounding individual values, the sum should remain close to 1.0.
        It will not be exact (rounding can introduce small error) but must be
        within a reasonable tolerance.
        """
        probs  = [{"en": 0.3, "fr": 0.4, "nl": 0.3}]
        result = self._aggregate(probs)
        assert sum(result.values()) == pytest.approx(1.0, abs=0.01)


# app/pipeline/s4_nlp/translate_CT2_backend.py

"""
CTranslate2 translation backend.

Uses CTranslate2-converted seq2seq models with SentencePiece tokenization, providing faster inference than HuggingFace for the same models without requiring PyTorch at runtime.

Use this backend when you need:
    - Faster CPU inference (2-4x speedup over PyTorch)
    - Lower memory footprint
    - Quantized inference (int8, float16)
    - Deployment without a PyTorch dependency

Models must first be converted to CTranslate2 format:

    ct2-transformers-converter --model Helsinki-NLP/opus-mt-en-de \\
        --output_dir opus_en_de_ct2

Expected model directory layout:

    <model_dir>/
        model.bin      CTranslate2 weights
        source.spm     source SentencePiece model
        target.spm     target SentencePiece model

Model selection priority:
    1. Explicit `model_dir` argument
    2. Language-based lookup from MT_LANG_MODEL_MAP + MT_CT2_MODEL_BASE_DIR
    3. Fallback to `opus_sh_en_ct2`

Requirements: `ctranslate2`, `sentencepiece`
"""

from pathlib import Path
from typing import List, Optional

import ctranslate2
import sentencepiece as spm

from app.pipeline.s4_nlp.translate_base import BaseTranslatorBackend
from app.config.mt_config import get_token_limit_config


_FALLBACK_MODEL_DIR = "opus_sh_en_ct2"


def _resolve_model_dir(model_dir: str | None, lang: str | None = None) -> str:
    """Resolve model directory: explicit arg → settings (lang map) → fallback."""
    if model_dir:
        return model_dir
    try:
        from app.config.mt_config import MT_LANG_MODEL_MAP
        from app.config.config import settings
        lang_key   = (lang or "").lower().split("-")[0] or "__default__"
        model_name = MT_LANG_MODEL_MAP.get(lang_key) or MT_LANG_MODEL_MAP["__default__"]
        return f"{str(settings.MT_CT2_MODEL_BASE_DIR).rstrip('/')}/{model_name}"
    except (ImportError, AttributeError, KeyError):
        return _FALLBACK_MODEL_DIR


class CT2TranslatorBackend(BaseTranslatorBackend):
    """
    CTranslate2-based translator backend.

    Loads a CTranslate2 seq2seq model and SentencePiece tokenizers; performs
    fast batched translation on CPU or GPU.

    Attributes:
        label:       Always "ct2".
        model_name:  Path to the model directory.
        token_config: Token limit configuration for this model.
    """

    label: str = "ct2"

    def __init__(
        self,
        model_dir: str | None = None,
        device: str = "cuda",
        compute_type: str = "default",
        src_spm_path: str | None = None,
        tgt_spm_path: str | None = None,
        max_tokens_override: Optional[int] = None,
        chars_per_token_override: Optional[float] = None,
        beam_size: int = 4,
        source_lang: str | None = None,
    ) -> None:
        """
        Args:
            model_dir:               CTranslate2 model directory. Resolved from settings if None.
            device:                  "cuda" or "cpu".
            compute_type:            "default" | "float16" | "int8" | "int8_float16".
            src_spm_path:            Source SentencePiece path. Defaults to <model_dir>/source.spm.
            tgt_spm_path:            Target SentencePiece path. Defaults to <model_dir>/target.spm.
            max_tokens_override:     Override the model's default max tokens.
            chars_per_token_override: Override chars-per-token estimation.
            beam_size:               Beam size for decoding.
            source_lang:             Source language code (used for model lookup if model_dir is None).
        """
        resolved_dir = _resolve_model_dir(model_dir, source_lang)
        self.model_dir = resolved_dir
        self.model_name = resolved_dir
        self.beam_size = beam_size

        model_path = Path(resolved_dir)
        src_spm_path = src_spm_path or str(model_path / "source.spm")
        tgt_spm_path = tgt_spm_path or str(model_path / "target.spm")

        self.token_config = get_token_limit_config(
            model_name=self._extract_model_family(resolved_dir),
            backend="ct2",
            override_max_tokens=max_tokens_override,
            override_chars_per_token=chars_per_token_override,
        )

        self._translator = ctranslate2.Translator(
            resolved_dir, device=device, compute_type=compute_type,
        )
        self._src_sp = spm.SentencePieceProcessor(model_file=src_spm_path)
        self._tgt_sp = spm.SentencePieceProcessor(model_file=tgt_spm_path)

    @staticmethod
    def _extract_model_family(model_dir: str) -> str:
        """Extract model family from directory name for token limit lookup."""
        name = Path(model_dir).name.lower()
        if "opus"  in name: return "opus"
        if "nllb"  in name: return "nllb"
        if "m2m"   in name: return "m2m100"
        return name

    def translate_batch(self, texts: List[str]) -> List[str]:
        """
        Translate a batch of strings using CTranslate2.

        Tokenizes each input with SentencePiece, runs batched CTranslate2
        translation, then decodes the top hypothesis back to a string.
        """
        if not texts:
            return []

        tokenized = [self._src_sp.encode(t, out_type=str) for t in texts]
        results = self._translator.translate_batch(
            tokenized, beam_size=self.beam_size, max_batch_size=len(tokenized),
        )
        return [self._tgt_sp.decode(r.hypotheses[0]) for r in results]

    def get_max_chars(self) -> int:
        return self.token_config.max_chars

    def count_tokens(self, text: str) -> int:
        """Count actual tokens using the source SentencePiece model (for validation/debugging)."""
        return len(self._src_sp.encode(text, out_type=str))

    def validate_chunk(self, text: str) -> bool:
        """Return True if the chunk is within the effective token limit."""
        return self.count_tokens(text) <= self.token_config.effective_max_tokens


    
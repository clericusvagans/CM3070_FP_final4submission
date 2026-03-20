# app/pipeline/s4_nlp/translate_gemma_backend.py

"""
TranslateGemma translation backend.

Uses Google's TranslateGemma models (based on Gemma 3) for high-quality multilingual translation.

IMPORANT:
TranslateGemma does not work with HuggingFace's pipeline("translation") because:
    1. It uses `AutoModelForImageTextToText` (multimodal architecture).
    2. It requires a chat template with explicit language codes.
    3. Input must be structured as messages, not plain text.

Available models:
    google/translategemma-4b-it    (~3 GB VRAM with 4-bit quantization)
    google/translategemma-12b-it   (~7 GB VRAM with 4-bit quantization)
    google/translategemma-27b-it   (~15 GB VRAM with 4-bit quantization)

Model selection priority:
    1. Explicit `model_name` argument
    2. settings.MT_GEMMA_MODEL
    3. Fallback to `google/translategemma-12b-it`

Note: `local_files_only=True` is set to support offline / firewalled
environments. Models must be pre-downloaded and cached beforehand.
See: https://huggingface.co/docs/transformers/installation

License: these models require accepting Google's license on HuggingFace.
Run `huggingface-cli login` and accept the license at the model page.

Requirements: `torch`, `transformers` (≥4.45.0), `accelerate`,
              `bitsandbytes` (for quantization).
"""

import time
from typing import Dict, List, Optional

import torch

from app.pipeline.s4_nlp.translate_base import BaseTranslatorBackend
from app.config.mt_config import get_token_limit_config
from app.utils.logger import logger


_FALLBACK_MODEL = "google/translategemma-4b-it"


def _resolve_model_name(model_name: str | None) -> str:
    """Resolve model name: explicit arg → settings → fallback."""
    if model_name:
        return model_name
    try:
        from app.config.config import settings
        return settings.MT_GEMMA_MODEL
    except (ImportError, AttributeError):
        return _FALLBACK_MODEL


# TranslateGemma uses ISO 639-1 codes; these map the internal codes directly.
LANG_CODE_MAP: Dict[str, str] = {
    "sr": "sr", "hr": "hr", "bs": "bs", "sl": "sl", "mk": "mk",
    "ru": "ru", "uk": "uk", "fr": "fr", "nl": "nl", "en": "en",
    "de": "de", "es": "es", "it": "it", "pt": "pt", "pl": "pl",
    "cs": "cs", "sk": "sk", "bg": "bg", "ro": "ro", "hu": "hu",
    "el": "el", "tr": "tr", "ar": "ar", "he": "he", "zh": "zh",
    "ja": "ja", "ko": "ko",
}


class TranslateGemmaBackend(BaseTranslatorBackend):
    """
    TranslateGemma-based translator backend.

    Loads the model and processor once at construction time. Translation uses
    the official chat template API: input is formatted as a structured message,
    and only the newly generated tokens are decoded.

    Attributes:
        label:        Always "gemma".
        model_name:   HuggingFace model identifier.
        token_config: Token limit configuration.
        source_lang:  Source language code (ISO 639-1).
        target_lang:  Target language code (ISO 639-1).
    """

    label: str = "gemma"

    def __init__(
        self,
        model_name: str | None = None,
        source_lang: str = "sr",
        target_lang: str = "en",
        device_map: str = "auto",
        dtype: Optional[torch.dtype] = None,
        max_tokens_override: Optional[int] = None,
        chars_per_token_override: Optional[float] = None,
        max_new_tokens: int = 1024,
        load_in_4bit: bool = False,
        load_in_8bit: bool = False,
    ) -> None:
        """
        Args:
            model_name:               HuggingFace model name. Resolved from settings if None.
            source_lang:              Source language code (e.g. "sr", "uk").
            target_lang:              Target language code (e.g. "en").
            device_map:               Device mapping strategy ("auto", "cuda:0", "cpu").
            dtype:                    Weight dtype. None = auto-detect (bfloat16 on CUDA).
            max_tokens_override:      Override model's default max input tokens.
            chars_per_token_override: Override chars-per-token estimation.
            max_new_tokens:           Maximum tokens to generate.
            load_in_4bit:             Use 4-bit quantization (requires bitsandbytes).
            load_in_8bit:             Use 8-bit quantization (requires bitsandbytes).
        """
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.model_name = _resolve_model_name(model_name)
        self.source_lang = LANG_CODE_MAP.get(source_lang, source_lang)
        self.target_lang = LANG_CODE_MAP.get(target_lang, target_lang)
        self.max_new_tokens = max_new_tokens

        self.token_config = get_token_limit_config(
            model_name=self.model_name,
            backend="gemma",
            override_max_tokens=max_tokens_override,
            override_chars_per_token=chars_per_token_override,
        )

        if dtype is None:
            dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

        model_kwargs: dict = {"device_map": device_map, "local_files_only": True}

        if load_in_4bit:
            from transformers import BitsAndBytesConfig
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        elif load_in_8bit:
            from transformers import BitsAndBytesConfig
            model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        else:
            model_kwargs["dtype"] = dtype

        logger.info(f"[TranslateGemma] Loading processor from {self.model_name}...")
        self._processor = AutoProcessor.from_pretrained(
            self.model_name, local_files_only=True,
        )

        logger.info(
            f"[TranslateGemma] Loading model from {self.model_name} | "
            f"device_map={device_map} 4bit={load_in_4bit} 8bit={load_in_8bit}"
        )
        t0 = time.perf_counter()
        self._model = AutoModelForImageTextToText.from_pretrained(self.model_name, **model_kwargs)
        self._device = next(self._model.parameters()).device
        self._dtype  = dtype
        logger.info(
            f"[TranslateGemma] Loaded in {time.perf_counter() - t0:.1f}s | "
            f"device={self._device} dtype={self._dtype} max_new_tokens={self.max_new_tokens}"
        )

    def set_languages(self, source_lang: str, target_lang: str) -> None:
        """Update source and target languages for subsequent translations."""
        self.source_lang = LANG_CODE_MAP.get(source_lang, source_lang)
        self.target_lang = LANG_CODE_MAP.get(target_lang, target_lang)

    def _build_message(self, text: str) -> List[dict]:
        """Build the chat message structure required by the TranslateGemma template."""
        return [{
            "role": "user",
            "content": [{
                "type": "text",
                "source_lang_code": self.source_lang,
                "target_lang_code": self.target_lang,
                "text": text,
            }],
        }]

    def translate_batch(self, texts: List[str]) -> List[str]:
        """
        Translate a batch of strings using TranslateGemma.

        TranslateGemma is processed one text at a time through the chat template.
        Batching efficiency is therefore achieved at the chunking level, not here.
        Only newly generated tokens (beyond the input length) are decoded.
        """
        if not texts:
            return []

        batch_size = len(texts)
        batch_start = time.perf_counter()
        logger.info(
            f"[TranslateGemma] Starting batch: {batch_size} chunk(s) | "
            f"{self.source_lang}→{self.target_lang}"
        )

        translations = []
        for idx, text in enumerate(texts, 1):
            t0 = time.perf_counter()
            preview = text[:80].replace("\n", " ") + ("..." if len(text) > 80 else "")

            inputs = self._processor.apply_chat_template(
                self._build_message(text),
                tokenize=True, add_generation_prompt=True,
                return_dict=True, return_tensors="pt",
            )
            inputs = {k: v.to(self._device) for k, v in inputs.items()}
            input_len = inputs["input_ids"].shape[1]

            logger.debug(
                f"[TranslateGemma] Chunk {idx}/{batch_size}: "
                f"{len(text)} chars → {input_len} tokens | \"{preview}\""
            )

            gen_t0 = time.perf_counter()
            with torch.inference_mode():
                output = self._model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
            gen_time = time.perf_counter() - gen_t0

            generated = output[0][input_len:]
            output_len = len(generated)
            translation = self._processor.decode(generated, skip_special_tokens=True)

            logger.info(
                f"[TranslateGemma] Chunk {idx}/{batch_size} done: "
                f"{input_len}→{output_len} tokens | "
                f"{gen_time:.2f}s gen ({output_len/gen_time:.1f} tok/s) | "
                f"{time.perf_counter()-t0:.2f}s total"
            )
            translations.append(translation.strip())

        elapsed = time.perf_counter() - batch_start
        logger.info(
            f"[TranslateGemma] Batch complete: {batch_size} chunk(s) in {elapsed:.2f}s "
            f"({elapsed/batch_size:.2f}s avg/chunk)"
        )
        return translations

    def translate_single(self, text: str) -> str:
        """Convenience wrapper: translate one text."""
        results = self.translate_batch([text])
        return results[0] if results else ""

    def get_max_chars(self) -> int:
        return self.token_config.max_chars

    def count_tokens(self, text: str) -> int:
        """Count tokens using the processor's tokenizer (for validation/debugging)."""
        return len(self._processor.tokenizer.encode(text, add_special_tokens=True))

    def validate_chunk(self, text: str) -> bool:
        """Return True if the chunk is within the effective token limit."""
        return self.count_tokens(text) <= self.token_config.effective_max_tokens

    def close(self) -> None:
        """
        Release GPU memory held by the model and processor.

        Moves the model to CPU, deletes both references, runs GC, and clears
        the PyTorch CUDA cache. The instance should not be used after this call.
        """
        import gc
        logger.info("[TranslateGemma] Releasing GPU memory...")

        try:
            if getattr(self, "_model", None) is not None:
                self._model.cpu()
                del self._model
                self._model = None
        except Exception as e:
            logger.warning(f"[TranslateGemma] Error moving model to CPU: {e}")

        try:
            if getattr(self, "_processor", None) is not None:
                del self._processor
                self._processor = None
        except Exception as e:
            logger.warning(f"[TranslateGemma] Error deleting processor: {e}")

        gc.collect()

        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                alloc   = torch.cuda.memory_allocated() / 1024 ** 2
                reserved = torch.cuda.memory_reserved()  / 1024 ** 2
                logger.info(
                    f"[TranslateGemma] GPU memory after cleanup: "
                    f"allocated={alloc:.0f}MB reserved={reserved:.0f}MB"
                )
        except Exception as e:
            logger.warning(f"[TranslateGemma] Error clearing CUDA cache: {e}")


# ─── Singleton management ───

_gemma_backend_instance: Optional[TranslateGemmaBackend] = None


def get_translategemma_backend(
    model_name: str | None = None,
    source_lang: str = "sr",
    target_lang: str = "en",
    **kwargs,
) -> TranslateGemmaBackend:
    """
    Get or create a singleton TranslateGemmaBackend instance.

    Created on first call and reused for subsequent calls. Languages can be
    updated via set_languages() if the existing instance is reused.
    """
    global _gemma_backend_instance

    if _gemma_backend_instance is None:
        _gemma_backend_instance = TranslateGemmaBackend(
            model_name=model_name,
            source_lang=source_lang,
            target_lang=target_lang,
            **kwargs,
        )
    else:
        _gemma_backend_instance.set_languages(source_lang, target_lang)

    return _gemma_backend_instance


def clear_translategemma_backend() -> None:
    """Clear the singleton instance and release GPU memory via close()."""
    global _gemma_backend_instance

    if _gemma_backend_instance is not None:
        try:
            _gemma_backend_instance.close()
        except Exception as e:
            logger.warning(f"[TranslateGemma] Error during close(): {e}")

    _gemma_backend_instance = None




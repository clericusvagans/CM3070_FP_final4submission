# app/config/mt_config.py

"""
Machine Translation configuration.

Provides language-to-model mappings, model token limits, script-aware tokenization parameters, and helper functions that read from the centralized settings object.

All environment-based configuration is accessed via `settings`; this module provides derived/computed configuration that does not belong in Settings itself.
"""

from dataclasses import dataclass

from app.config.config import settings


# ─── Language → Model mappings ───
# Keys are primary_lang codes as set by S2 detection.

# CT2: value is a model directory name under MT_CT2_MODEL_BASE_DIR.
MT_LANG_MODEL_MAP: dict[str, str] = {
    "sr": "opus_sh_en_ct2", # Serbian (same model covers BHS)
    "bs": "opus_sh_en_ct2", # Bosnian
    "hr": "opus_sh_en_ct2", # Croatian
    "sl": "opus_sl_en_ct2", # Slovenian
    "mk": "opus_mk_en_ct2", # Macedonian
    "fr": "opus_fr_en_ct2", # French
    "nl": "opus_nl_en_ct2", # Dutch
    "ru": "opus_ru_en_ct2", # Russian
    "uk": "opus_uk_en_ct2", # Ukrainian
    "__default__": "opus_fr_en_ct2",
}

# HF: value is a HuggingFace model name or local path.
MT_HF_LANG_MODEL_MAP: dict[str, str] = {
    "sr": "Helsinki-NLP/opus-mt-tc-big-sh-en",
    "bs": "Helsinki-NLP/opus-mt-tc-big-sh-en",
    "hr": "Helsinki-NLP/opus-mt-tc-big-sh-en",
    "sl": "Helsinki-NLP/opus-mt-sl-en",
    "mk": "Helsinki-NLP/opus-mt-mk-en",
    "fr": "Helsinki-NLP/opus-mt-tc-big-en-fr",
    "nl": "Helsinki-NLP/opus-mt-nl-en",
    "ru": "Helsinki-NLP/opus-mt-ru-en",
    "uk": "Helsinki-NLP/opus-mt-uk-en",
    "__default__": "Helsinki-NLP/opus-mt-tc-big-en-fr",
}

# vLLM: value is a model identifier as served by the vLLM server.
MT_VLLM_LANG_MODEL_MAP: dict[str, str] = {
    "__default__": "facebook/nllb-200-distilled-600M",
}

# Note: TranslateGemma uses no language→model map; it is multilingual (one model
# for all languages) so mt_stage.py uses settings.MT_GEMMA_MODEL directly.


# ─── Token limit configuration ───

@dataclass(frozen=True)
class TokenLimitConfig:
    """
    Token limit and character-estimation parameters for a model family.

    Effective chunking limit:
        max_chars = (max_tokens - safety_margin) * chars_per_token

    Example (OPUS-MT defaults):
        max_chars = (512 - 62) * 3.5 = 1575 characters
    """
    max_tokens: int = 512
    safety_margin: int = 62
    chars_per_token: float = 3.5

    @property
    def effective_max_tokens(self) -> int:
        """Maximum tokens after applying safety margin."""
        return self.max_tokens - self.safety_margin

    @property
    def max_chars(self) -> int:
        """Maximum characters per chunk based on token estimation."""
        return int(self.effective_max_tokens * self.chars_per_token)


# ─── Model token limit registry ───
# Keys: model family prefix ("opus"), exact model name, backend type ("vllm"),
# or "__default__" fallback.

MODEL_TOKEN_LIMITS: dict[str, TokenLimitConfig] = {
    "opus": TokenLimitConfig(max_tokens=512,  safety_margin=62,  chars_per_token=3.5),

    # TranslateGemma: 2K token INPUT context per Google documentation.
    # Note: "translategemma" must be checked before "gemma" in prefix matching
    # because "google/translategemma-12b-it" contains both substrings.
    "translategemma": TokenLimitConfig(max_tokens=2048, safety_margin=200, chars_per_token=4.0),
    "gemma":          TokenLimitConfig(max_tokens=2048, safety_margin=200, chars_per_token=4.0),

    "nllb":   TokenLimitConfig(max_tokens=1024, safety_margin=74,  chars_per_token=4.0),
    "m2m100": TokenLimitConfig(max_tokens=1024, safety_margin=74,  chars_per_token=4.0),
    "mbart":  TokenLimitConfig(max_tokens=1024, safety_margin=74,  chars_per_token=4.0),
    "vllm":   TokenLimitConfig(max_tokens=2048, safety_margin=128, chars_per_token=4.0),

    "__default__": TokenLimitConfig(max_tokens=512, safety_margin=62, chars_per_token=3.5),
}


# ── Script-aware character estimation ────────────────────────────────────────
# Different scripts have different characters-per-token ratios due to
# SentencePiece tokenization. Cyrillic produces more tokens per character
# than Latin, so the limit is more conservative.

SCRIPT_CHARS_PER_TOKEN: dict[str, float] = {
    "Latn": 4.0,  # Latin
    "Cyrl": 3.0,  # Cyrillic (more conservative)
    "Arab": 3.5,
    "Grek": 3.5,
    "Hans": 1.5,  # Simplified Chinese
    "Hant": 1.5,  # Traditional Chinese
    "Jpan": 2.0,
    "Kore": 2.0,
    "__default__": 3.5,
}


def get_chars_per_token_for_script(script: str | None) -> float:
    """Return the character-to-token ratio for the given ISO 15924 script code."""
    if not script:
        return SCRIPT_CHARS_PER_TOKEN["__default__"]
    return SCRIPT_CHARS_PER_TOKEN.get(script, SCRIPT_CHARS_PER_TOKEN["__default__"])


# ───── Token limit resolution ───

def get_token_limit_config(
    model_name: str,
    backend: str = "hf",
    override_max_tokens: int | None = None,
    override_chars_per_token: float | None = None,
) -> TokenLimitConfig:
    """
    Resolve the token limit configuration for a given model.

    Resolution order:
        1. Explicit override parameters (if provided).
        2. Exact model name match in MODEL_TOKEN_LIMITS.
        3. Model family prefix match ("opus", "translategemma", "gemma", …).
           IMPORTANT: more specific patterns come first — "translategemma"
           must be checked before "gemma" because model names like
           "google/translategemma-12b-it" contain both substrings.
        4. Backend-specific default ("vllm", etc.).
        5. Global "__default__" fallback.

    Args:
        model_name:                Full model identifier or directory path.
        backend:                   Backend type ("ct2", "hf", "vllm", "gemma").
        override_max_tokens:       Explicit max tokens override.
        override_chars_per_token:  Explicit chars-per-token override.

    Returns:
        Resolved TokenLimitConfig.
    """
    config = MODEL_TOKEN_LIMITS["__default__"]
    model_lower = model_name.lower()

    if model_name in MODEL_TOKEN_LIMITS:
        config = MODEL_TOKEN_LIMITS[model_name]
    else:
        matched = False
        for pattern in ["opus", "translategemma", "gemma", "nllb", "m2m100", "mbart"]:
            if pattern in model_lower and pattern in MODEL_TOKEN_LIMITS:
                config = MODEL_TOKEN_LIMITS[pattern]
                matched = True
                break
        if not matched and backend in MODEL_TOKEN_LIMITS:
            config = MODEL_TOKEN_LIMITS[backend]

    if override_max_tokens is not None or override_chars_per_token is not None:
        config = TokenLimitConfig(
            max_tokens = override_max_tokens if override_max_tokens is not None else config.max_tokens,
            safety_margin = config.safety_margin,
            chars_per_token = override_chars_per_token if override_chars_per_token is not None else config.chars_per_token,
        )

    return config


# ─── Settings access helpers ───

def get_backend()                    -> str:          return settings.MT_BACKEND
def get_batch_size()                 -> int:          return settings.MT_BATCH_SIZE
def get_manual_token_limit_override()-> int | None:   return settings.MT_MAX_TOKENS_OVERRIDE
def get_chars_per_token_override()   -> float | None: return settings.MT_CHARS_PER_TOKEN_OVERRIDE
def is_debug_chunks_enabled()        -> bool:         return settings.DEBUG_MT_CHUNKS


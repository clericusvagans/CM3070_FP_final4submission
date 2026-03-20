# app/pipeline/s4_nlp/translate_base.py

"""
Abstract translator interface and text chunking utilities.

This module is backend-agnostic and defines the common interface and utilities shared by all translation backends.

Classes
-------
BaseTranslatorBackend   Abstract interface every backend must implement.
TextChunker             Token-aware chunker that preserves document structure.
_ChunkAccumulator       Internal helper for consistent separator accounting.

Functions
---------
translate_with_chunking Chunk → translate → reassemble.
batched                 Split an iterable into fixed-size batches.

Concrete backends live in sibling modules:
    translate_CT2_backend.py     CTranslate2 + SentencePiece (fast inference)
    translate_gemma_backend.py   TranslateGemma (chat-based API)
    translate_HF_backend.py      HuggingFace pipeline (OPUS, NLLB, M2M)
    translate_vLLM_backend.py    vLLM inference server

Token limit handling
--------------------
The chunking algorithm is designed to never exceed the configured character
limit (which maps to the model's token limit), preserve structure
(paragraphs > sentences > words), and handle edge cases such as very long
words and mixed scripts.

Key design: separator length is accounted for *before* deciding whether to
add an item to the accumulator. This prevents off-by-one errors that could
cause token limit violations.
"""

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Iterable, Iterator


# ─── Abstract backend interface ───

class BaseTranslatorBackend(ABC):
    """
    Abstract interface for translation backends.

    Any concrete backend must implement `translate_batch()` and set
    `token_config` in `__init__`.

    Attributes:
        label:       Short backend identifier (e.g. "ct2", "gemma").
        model_name:  Full model identifier or path.
        token_config: Token limit configuration — must be set by subclass.
    """

    label: str = "base"
    model_name: str = ""

    @abstractmethod
    def translate_batch(self, texts: list[str]) -> list[str]:
        """
        Translate a batch of input strings.

        Args:
            texts: list of input strings.

        Returns:
            list of translated strings, same length as `texts`.
        """
        raise NotImplementedError

    def get_max_chars(self) -> int:
        """Return the maximum characters per chunk for this backend."""
        return self.token_config.max_chars


# ─── Batching utility ───

def batched(iterable: Iterable[str], batch_size: int) -> Iterator[list[str]]:
    """Split an iterable of strings into batches of at most `batch_size` items."""
    batch: list[str] = []
    for item in iterable:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


# ─── Text chunking ───

# Sentence boundary: after .!? + space + capital letter, or at line ends.
_SENTENCE_SPLIT_RE = re.compile(
    r'(?<=[.!?])\s+(?=[A-Z\u0400-\u04FF\u0100-\u017F])|'
    r'(?<=[.!?])\s*$|'
    r'(?<=\n)\s*'
)


@dataclass
class _ChunkAccumulator:
    """
    Internal accumulator for building chunks with consistent separator handling.

    Separator length is calculated *before* deciding whether to add an item,
    preventing off-by-one errors that could cause token limit violations.
    """
    max_chars: int
    separator: str
    _items: list[str] = field(default_factory=list)
    _current_len: int = 0

    def try_add(self, item: str) -> bool:
        """Try to add an item. Returns True on success, False if it would exceed the limit."""
        sep_len = len(self.separator) if self._items else 0
        new_total = self._current_len + sep_len + len(item)
        if new_total <= self.max_chars:
            self._items.append(item)
            self._current_len = new_total
            return True
        return False

    def flush(self) -> str:
        """Return the joined chunk and reset the accumulator. Returns "" if empty."""
        if not self._items:
            return ""
        result = self.separator.join(self._items)
        self._items = []
        self._current_len = 0
        return result


@dataclass
class TextChunker:
    """
    Token-aware text chunker that preserves document structure.

    Splits text into chunks guaranteed to be at most `max_chars` characters.
    Strategy: paragraphs → sentences → words → hard split (last resort).

    Attributes:
        max_chars:           Maximum characters per chunk (default 1575 ≈ 450 tokens × 3.5).
        paragraph_separator: Separator used between paragraphs.
        sentence_separator:  Separator used between sentences.
    """
    max_chars: int = 1575
    paragraph_separator: str = "\n\n"
    sentence_separator: str = " "

    def chunk(self, text: str) -> list[str]:
        """
        Split text into chunks within the character limit.

        1. Split by paragraphs (\\n\\n) — natural boundaries.
        2. Short paragraphs are accumulated; long ones are sentence-split.
        3. Long sentences are word-split; words exceeding the limit are hard-split.
        """
        if not text or not text.strip():
            return []
        text = text.strip()
        if len(text) <= self.max_chars:
            return [text]

        paragraphs = [p.strip() for p in text.split(self.paragraph_separator) if p.strip()]
        final_chunks: list[str] = []
        acc = _ChunkAccumulator(max_chars=self.max_chars, separator=self.paragraph_separator)

        for para in paragraphs:
            if len(para) <= self.max_chars:
                if not acc.try_add(para):
                    if chunk := acc.flush():
                        final_chunks.append(chunk)
                    acc.try_add(para)
            else:
                if chunk := acc.flush():
                    final_chunks.append(chunk)
                final_chunks.extend(self._chunk_paragraph(para))

        if chunk := acc.flush():
            final_chunks.append(chunk)
        return final_chunks

    def _chunk_paragraph(self, paragraph: str) -> list[str]:
        """Split a long paragraph into sentence-based chunks."""
        sentences = self._split_into_sentences(paragraph)
        chunks: list[str] = []
        acc = _ChunkAccumulator(max_chars=self.max_chars, separator=self.sentence_separator)

        for sentence in sentences:
            if len(sentence) <= self.max_chars:
                if not acc.try_add(sentence):
                    if chunk := acc.flush():
                        chunks.append(chunk)
                    acc.try_add(sentence)
            else:
                if chunk := acc.flush():
                    chunks.append(chunk)
                chunks.extend(self._chunk_by_words(sentence))

        if chunk := acc.flush():
            chunks.append(chunk)
        return chunks

    def _chunk_by_words(self, text: str) -> list[str]:
        """Split text at word boundaries; hard-split words that exceed the limit."""
        chunks: list[str] = []
        acc = _ChunkAccumulator(max_chars=self.max_chars, separator=" ")

        for word in text.split():
            if len(word) <= self.max_chars:
                if not acc.try_add(word):
                    if chunk := acc.flush():
                        chunks.append(chunk)
                    acc.try_add(word)
            else:
                if chunk := acc.flush():
                    chunks.append(chunk)
                for i in range(0, len(word), self.max_chars):
                    chunks.append(word[i: i + self.max_chars])

        if chunk := acc.flush():
            chunks.append(chunk)
        return chunks

    def _split_into_sentences(self, text: str) -> list[str]:
        """Split text into sentences using the sentence boundary regex."""
        if not text.strip():
            return []
        sentences = [s.strip() for s in _SENTENCE_SPLIT_RE.split(text) if s and s.strip()]
        return sentences or [text.strip()]


# ─── High-level translation helper ───

def translate_with_chunking(
    backend: BaseTranslatorBackend,
    text: str,
    batch_size: int = 6,
    max_chars_override: int | None = None,
) -> str:
    """
    Translate text that may exceed the model's token limit.

    Chunks the text, translates each chunk in batches, and reassembles with
    paragraph separators.

    Args:
        backend:            Translation backend instance.
        text:               Full text to translate.
        batch_size:         Chunks per batch call.
        max_chars_override: Override the backend's default max_chars.

    Returns:
        Translated text with structure preserved.
    """
    if not text or not text.strip():
        return ""

    max_chars = max_chars_override if max_chars_override is not None else backend.get_max_chars()
    chunks = TextChunker(max_chars=max_chars).chunk(text)
    if not chunks:
        return ""

    translated: list[str] = []
    for batch in batched(chunks, batch_size):
        translated.extend(backend.translate_batch(batch))

    return "\n\n".join(t.strip() for t in translated if t and t.strip())



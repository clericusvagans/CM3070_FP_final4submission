# app/models.py

"""
SQLAlchemy ORM models.

Schema overview
---------------
Built around PaddleOCR / PP-StructureV3, where the canonical content unit is
a layout block (PageBlock).

Models
------
    DuplicateGroup  Content-identity group; one row per unique file hash.
    Document        Top-level container: provenance, status, language metadata.
    Page            Per-page representation with dimensions, DPI, and parsed content.
    PageBlock       Layout regions (Text/Title/Table/Picture/…) with geometry + OCR text.
    PageMT          Machine translation of a full page's text_layer (one row per target lang).
    BlockMT         Machine translation of an individual block (one row per block/lang).
    PageImage       Extracted images/figures from pages.
    Entity          NER mentions extracted from document text.
    Job             Background job tracking for pipeline stages.

Coordinate conventions
----------------------
All geometry is in pixel coordinates of the *persisted page image* produced at
ingestion time. This avoids PDF-point vs pixel confusion.

Page numbering
--------------
Page.page_number is 1-based (first page = 1).

Content fields
--------------
    Page.markdown       Display-oriented markdown; may omit text from Picture regions.
    Page.text_layer     Concatenation of PageBlock.text in reading order; MT input.
    PageMT.text         Whole-page translation (from Page.text_layer, not markdown).
    BlockMT.text        Per-block translation.

Full-text search
----------------
GIN indexes on `to_tsvector()` expressions are declared for
`Page.markdown`, `PageMT.text`, and `BlockMT.text`.

JSONB dual-layer defaults
-------------------------
Several JSONB / ARRAY columns carry both a Python-side ORM default and a
server-side DB default:

    default=dict / default=list
        Called by SQLAlchemy before INSERT. Only active for ORM-originated inserts.

    server_default=text("'{}' ::jsonb") / server_default=text("'{}' ::text[]")
        DDL-level DEFAULT clause evaluated by Postgres for every INSERT that
        omits the column — including raw psql, Alembic data migrations, fixtures,
        and any non-ORM tool. Without this, those inserts produce NULL, which
        causes crashes in pipeline code that calls .items(), .get(), or iterates
        over the field without a None guard.

        Note: Postgres array literal syntax uses `'{}'`, not `'[]'` (JSON).
              The cast `::text[]` tells Postgres which array type to use.
"""


from sqlalchemy import (
    BigInteger, Boolean, CheckConstraint, Column, DateTime, Enum, Float,
    ForeignKey, Index, Integer, LargeBinary, String, Text, UniqueConstraint,
    func,
    text as sql_text, # SQL expression constructor — distinct from the Text column type above
)
from sqlalchemy.dialects.postgresql import UUID, JSONB, ARRAY
from sqlalchemy.orm import relationship
from datetime import datetime
from app.db import Base
import uuid, enum


# ─── Enums ───

class DocStatus(str, enum.Enum):
    """
    Document-level pipeline lifecycle status.

    Per-stage job state lives in Job; per-page/parsing metadata lives in
    Page.page_metadata.
    """
    queued = "queued"
    processing = "processing"
    completed = "completed"
    failed = "failed"
    deleting = "deleting" # soft-delete gate — workers must abort on this status
    deleted = "deleted"


# ─── DuplicateGroup ───

class DuplicateGroup(Base):
    """
    Canonical content-identity group for duplicate tracking.

    Several Document rows may point to the same content hash, particularly when
    the API is called with mode=0 ("always create a new row even if bytes already
    exist"). The normalized model is:

        DuplicateGroup  1 ──── *  Document

    Lifecycle: DuplicateGroup rows are never deleted automatically. If the last
    Document referencing a group is deleted, the group row becomes an intentional
    orphan (tombstone). The FK uses `ondelete="RESTRICT"` to prevent accidental
    cascade. Periodic garbage collection may remove orphaned groups if desired.

    Natural key: (hash_algo, content_hash) — enforced via unique constraint.
    Surrogate integer PK is used for join ergonomics.
    """
    __tablename__ = "duplicate_groups"

    id = Column(Integer, primary_key=True)
    hash_algo = Column(String(16), nullable=False)
    content_hash = Column(String(64), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    documents = relationship("Document", back_populates="duplicate_group")

    __table_args__ = (
        UniqueConstraint("hash_algo", "content_hash",
                         name="uq_duplicate_groups_hash_algo_content_hash"),
        Index("ix_duplicate_groups_hash_lookup", "hash_algo", "content_hash"),
        CheckConstraint("hash_algo IN ('sha256')", name="ck_duplicate_groups_hash_algo_allowed"),
    )


# ─── Document ───

class Document(Base):
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, index=True)

    # Documents are not unique by hash. Multiple Document rows may represent
    # identical bytes (e.g. mode=0 "always create"). Each points to one
    # DuplicateGroup; the group is the canonical content-identity record.
    duplicate_group_id = Column(
        Integer, ForeignKey("duplicate_groups.id", ondelete="RESTRICT"),
        nullable=False, index=True,
    )

    hash_algo = Column(String(16), nullable=False, server_default="sha256")
    content_hash = Column(String(64), nullable=False, index=True)
    byte_size = Column(BigInteger, nullable=False)

    filename = Column(String(255), nullable=False)
    source_uri = Column(String, nullable=False)   # output:// URI to SOURCEFILE
    dpi = Column(Integer, nullable=True)

    status = Column(
        Enum(DocStatus, name="doc_status"),
        nullable=False, default=DocStatus.queued,
    )

    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(),
                        onupdate=func.now(), nullable=False)

    # General-purpose document metadata (pipeline run info, artifact roots,
    # feature flags). See module docstring for dual-layer default rationale.
    doc_metadata = Column(
        JSONB,
        default=dict,
        server_default=sql_text("'{}' ::jsonb"),
    )

    # Language detection (document-level).
    # Postgres array literal uses '{}' (not '[]' which is JSON syntax).
    # Cast ::text[] tells Postgres the array element type.
    declared_langs = Column(
        ARRAY(String),
        default=list,
        server_default=sql_text("'{}' ::text[]"),
    )

    detected_langs = Column(
        JSONB,
        default=dict,
        server_default=sql_text("'{}' ::jsonb"),
    )   # e.g. {"en": 0.8, "fr": 0.2}

    primary_lang = Column(String, nullable=True)
    primary_script = Column(String, nullable=True)

    # Page-level source type: "image" for image inputs;
    # "native-pdf" | "image-pdf" | "ocr-pdf" for PDFs.
    sourcetype = Column(String, nullable=True)

    duplicate_group = relationship("DuplicateGroup", back_populates="documents")
    pages = relationship("Page", back_populates="document", cascade="all, delete-orphan")
    entities = relationship("Entity", back_populates="document", cascade="all, delete-orphan")
    jobs = relationship("Job", back_populates="document", cascade="all, delete-orphan")

    __table_args__ = (
        Index("ix_documents_status_created", "status", "created_at"),
        # Hash uniqueness constraint was removed: multiple Documents may share a hash
        # (linked to one DuplicateGroup). This index replaces it for fast hash lookups.
        Index("ix_documents_hash_lookup", "hash_algo", "content_hash"),
        CheckConstraint("hash_algo IN ('sha256')", name="ck_documents_hash_algo_allowed"),
    )


# ─── Page ───

class Page(Base):
    __tablename__ = "pages"

    id = Column(Integer, primary_key=True)
    document_id = Column(Integer, ForeignKey("documents.id", ondelete="CASCADE"),
                         index=True, nullable=False)
    page_number = Column(Integer, nullable=False) # 1-based

    # Dimensions of the rasterized page image produced at ingestion.
    width = Column(Integer)
    height = Column(Integer)

    # DPI used to materialize the page image.
    # For PDFs: applied at render time (may vary per page via pixel-budget heuristics).
    # For raster inputs: inferred from metadata or heuristics (pixels not resampled).
    dpi = Column(Integer, nullable=True)

    # Language info (page-level). See dual-layer default rationale in module docstring.
    lang_probs = Column(JSONB, default=dict,
                            server_default=sql_text("'{}'::jsonb"))  # e.g. {"en": 0.6, "sr-Cyrl": 0.4}
    primary_lang = Column(String, nullable=True)
    primary_script = Column(String, nullable=True)
    # NOTE: ocr_lang_hint was removed. The authoritative candidate language list lives
    # in Document.declared_langs and is shared uniformly across all pages.

    sourcetype = Column(String, nullable=True)

    # Page metadata container (most heavily accessed JSONB field).
    # Written by every stage: S0 writes image_uri, S3 writes "paddle" dict, etc.
    # Not named `metadata` — that is a reserved attribute on SQLAlchemy declarative models.
    page_metadata = Column(JSONB, default=dict, server_default=sql_text("'{}' ::jsonb"))

    # Canonical MT input: best-effort concatenation of PageBlock.text in reading order
    # (reading_order preferred; block_index as fallback).
    # Page.markdown is display-oriented and may omit text from Picture/Figure regions.
    text_layer = Column(Text, nullable=True)

    # md_uri: output:// URI of the .md file produced by S3 parsing.
    # markdown: Full content of the parsed markdown page stored directly in DB to
    #           enable Postgres FTS via the GIN-indexed tsvector expression below.
    #           Kept as TEXT (not TSVECTOR) so it is human-readable and can be
    #           re-tokenized with any dictionary at query time.
    md_uri = Column(String, nullable=True)
    markdown = Column(Text, nullable=True)

    document = relationship("Document", back_populates="pages")
    blocks = relationship("PageBlock", back_populates="page", cascade="all, delete-orphan")
    translations = relationship("PageMT", back_populates="page", cascade="all, delete-orphan")
    images = relationship("PageImage", back_populates="page", cascade="all, delete-orphan")

    __table_args__ = (
        UniqueConstraint("document_id", "page_number", name="uq_pages_document_id_index"),
        Index("ix_pages_document_id", "document_id"),
        # GIN FTS index on markdown.
        # Why 'simple' dictionary: documents may be in multiple languages; 'simple'
        # (no stemming, no stop-words) is a safe universal default. Callers may
        # switch to a language-specific dictionary at query time as long as they
        # use the same dictionary here and in the WHERE clause.
        Index(
            "ix_pages_markdown_fts",
            sql_text("to_tsvector('simple', coalesce(markdown, ''))"),
            postgresql_using="gin",
        ),
    )


# ─── PageBlock ───

class PageBlock(Base):
    """
    Canonical parsed unit for a Page.

    A layout region (Text/Title/Table/Picture/Header/Footer/…) with recognized
    text and geometry, produced by PP-StructureV3 or similar.

    Geometry: bbox_px = [x1, y1, x2, y2] in pixels of the page image (required).
    Ordering: block_index (0-based, stable); reading_order (explicit, may be None).
    """
    __tablename__ = "page_blocks"

    id = Column(Integer, primary_key=True)
    page_id = Column(Integer, ForeignKey("pages.id", ondelete="CASCADE"),
                     index=True, nullable=False)

    block_index = Column(Integer, nullable=False)  # 0..N-1 (parsing_res_list index)
    reading_order = Column(Integer, nullable=True)

    # Layout category (normalize at the adapter layer).
    # Examples: "Text" "Title" "Table" "Picture" "Header" "Footer" "Seal" "Formula" "List"
    category = Column(String(64), nullable=False)

    bbox_px = Column(JSONB, nullable=False)  # [x1, y1, x2, y2] in page-image pixels
    text = Column(Text, default="")

    # Which engine produced this block:
    # "pp_structure_v3" | "paddle_vl_1_5" | "pp_ocr_v5" | …
    engine = Column(String(64), nullable=True)

    # Extra structured data (block_id, table HTML, formula LaTeX, etc.)
    extra = Column(JSONB, default=dict, server_default=sql_text("'{}'::jsonb"))

    page = relationship("Page", back_populates="blocks")
    translations = relationship("BlockMT", back_populates="block", cascade="all, delete-orphan")

    __table_args__ = (
        UniqueConstraint("page_id", "block_index", name="uq_page_blocks_page_id_block_index"),
        Index("ix_page_blocks_page_id", "page_id"),
        Index("ix_page_blocks_page_id_category", "page_id", "category"),
        Index(
            "ix_page_blocks_text_fts",
            sql_text("to_tsvector('simple', coalesce(text, ''))"),
            postgresql_using="gin",
        ),
    )


# ─── Translations ───

class PageMT(Base):
    """
    Machine translation of a Page's text_layer into a target language.
    One row per (page, target_language) pair.

    Translated from Page.text_layer (whole-page, not markdown) to avoid quality
    degradation from segmentation artifacts.
    """
    __tablename__ = "page_mt"

    id = Column(Integer, primary_key=True)
    page_id = Column(Integer, ForeignKey("pages.id", ondelete="CASCADE"),
                     index=True, nullable=False)

    lang = Column(String(16), nullable=False)  # ISO 639-1 / BCP-47, e.g. "en"
    text = Column(Text, nullable=False)
    uri = Column(String, nullable=True)  # output:// URI to translated .md file
    translated_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    model = Column(String(128), nullable=False)  # e.g. "ct2" or "ct2:opus_uk_en_ct2"

    page = relationship("Page", back_populates="translations")

    __table_args__ = (
        UniqueConstraint("page_id", "lang", name="uq_page_mt_page_id_lang"),
        Index("ix_page_mt_page_id", "page_id"),
        Index(
            "ix_page_mt_text_fts",
            sql_text("to_tsvector('simple', coalesce(text, ''))"),
            postgresql_using="gin",
        ),
    )


class BlockMT(Base):
    """
    Machine translation of a PageBlock's text into a target language.
    One row per (block, target_language) pair.
    Block-level translations enable granular retrieval and alignment.
    """
    __tablename__ = "block_mt"

    id = Column(Integer, primary_key=True)
    block_id = Column(Integer, ForeignKey("page_blocks.id", ondelete="CASCADE"),
                      index=True, nullable=False)

    lang = Column(String(16), nullable=False)
    text = Column(Text, nullable=False)
    translated_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    model = Column(String(128), nullable=False)

    block = relationship("PageBlock", back_populates="translations")

    __table_args__ = (
        UniqueConstraint("block_id", "lang", name="uq_block_mt_block_id_lang"),
        Index("ix_block_mt_block_id", "block_id"),
        Index(
            "ix_block_mt_text_fts",
            sql_text("to_tsvector('simple', coalesce(text, ''))"),
            postgresql_using="gin",
        ),
    )


# ─── PageImage ───

class PageImage(Base):
    """
    Extracted images/figures from pages (charts, tables as images).

    Sourced from `res.markdown["markdown_images"]` in PP-StructureV3 output:
    `{"imgs/table_0.png": <PIL.Image>, …}`

    Storage strategy:
        filename:  Relative path key from markdown_images (e.g. "imgs/table_0.png");
                  used to resolve markdown image references back to this row.
        uri:       Always populated — output:// URI to the saved image file.
        data:      Optional BYTEA blob for direct DB retrieval without filesystem access.
    """
    __tablename__ = "images"

    id = Column(Integer, primary_key=True)
    page_id = Column(Integer, ForeignKey("pages.id", ondelete="CASCADE"),
                     index=True, nullable=False)

    filename = Column(String(512), nullable=False)  # markdown-relative path key
    uri = Column(String, nullable=False)  # output:// URI
    data = Column(LargeBinary, nullable=True)  # optional BYTEA blob

    page = relationship("Page", back_populates="images")

    __table_args__ = (
        Index("ix_page_images_page_id", "page_id"),
        Index("ix_page_images_filename", "filename"),
        UniqueConstraint("page_id", "filename", name="uq_page_images_page_id_filename"),
    )


# ─── Entity ───

class Entity(Base):
    """
    NER output — named entity mentions in document text.
    (PERSON / ORG / DATE / MONEY / LAW / CASE_ID / …)

    start/end are offsets in the canonical page text representation (text_layer).
    May be extended with page_id/block_id for block-level alignment.
    """
    __tablename__ = "entities"

    id = Column(Integer, primary_key=True)
    document_id = Column(Integer, ForeignKey("documents.id", ondelete="CASCADE"),
                         nullable=False, index=True)

    label = Column(String(64),  nullable=False)    # entity type
    text = Column(String(255), nullable=False)    # surface form
    start = Column(Integer,     nullable=False)    # offset in text_layer
    end = Column(Integer,     nullable=False)

    document = relationship("Document", back_populates="entities")

    __table_args__ = (
        Index("ix_entities_document_id", "document_id"),
        Index("ix_entities_label", "label"),
    )


# ─── Job ───

class Job(Base):
    """
    Persistent job/run tracking for background processing.

    Job.id is the Redis RQ job ID, correlating queue/run state in Redis with
    durable logs and status in Postgres.

    log schema: {"events": [{"ts": "…", "level": "…", "stage": "…", "message": "…"}, …]}
    """
    __tablename__ = "jobs"

    id = Column(String, primary_key=True)  # Redis RQ job ID

    document_id = Column(Integer, ForeignKey("documents.id", ondelete="CASCADE"),
                         index=True, nullable=False)

    stage = Column(String, nullable=True)  # pipeline | s0_materialize | s3_parse | …
    status = Column(String, default="queued")  # queued | started | finished | failed

    log = Column(
        JSONB,
        default=lambda: {"events": []},
        server_default=sql_text('\'{"events": []}\'::jsonb'),
    )

    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(),
                        onupdate=func.now(), nullable=False)

    document = relationship("Document", back_populates="jobs")

    __table_args__ = (
        Index("ix_jobs_document_id", "document_id"),
        Index("ix_jobs_status", "status"),
    )



# app/pipeline/orchestrator_std.py

"""
Standard (DB-backed) pipeline orchestrator executed by RQ workers.

Overview
--------
FastAPI routes ingest the raw upload and enqueue a job by dotted string
reference. This orchestrator runs inside the RQ worker process and owns:
    - stage sequencing (S0-S5)
    - transaction boundaries (one commit per stage)
    - job tracking events (append_event / set_status)
    - per-document stage summary in doc.doc_metadata["stages"]

Pipeline stages
---------------
    S0  Materialise pages + page images (mandatory — failure aborts the job).
    S1  Classify input: sourcetype (native-pdf / image-pdf / ocr-pdf / image).
    S2  Detection: language and script detection.
    S3  Parse (DLA/OCR via Paddle PP-StructureV3) + PageBlocks.
    S4a NLP: Machine Translation (page text_layer → target language).
    S5  Embeddings / vector indexing (not yet implemented).

Stage failure policy
--------------------
    S0 is mandatory: any exception aborts the job.
    S1-S5 are optional:
        import errors  → stage marked "skipped", pipeline continues.
        runtime errors → stage marked "failed", pipeline continues.
    (A strict=True mode that aborts on any stage failure is not yet implemented.)

Auditability
------------
Two complementary layers:
    Job tracking events         Operational trace keyed by job_id.
    doc_metadata["stages"]      Compact per-document summary for UI/debug
                                inspection without a join on the jobs table.

doc_metadata["stages"] convention
----------------------------------
    {"s3_parse": {"status": "ok" | "failed" | "skipped", "detail": "…"}}
"""

from typing import Any

from rq import get_current_job
from sqlalchemy.orm import Session

from app.db import get_sync_session
from app.models import Document, DocStatus, Page
from app.pipeline.modes import PipelineMode, parse_pipeline_mode
from app.utils.job_tracking import append_event, ensure_job_row, set_status
from app.utils.logger import logger


# ── DB helpers ────────────────────────────────────────────────────────────────

def _load_doc_or_fail(db: Session, doc_id: int) -> Document:
    """
    Load a Document and verify it is in a processable state.
    Raises RuntimeError if missing or already deleting/deleted.
    """
    doc = db.get(Document, doc_id)
    if doc is None:
        raise RuntimeError(f"Document not found doc_id={doc_id}")
    if doc.status in (DocStatus.deleting, DocStatus.deleted):
        raise RuntimeError(
            f"Document is deleted/deleting doc_id={doc_id} status={doc.status}"
        )
    return doc


def _refresh_pages(db: Session, doc_id: int) -> list[Page]:
    """
    Reload pages for a document in page-number order.
    Should be called after any stage commit to keep the in-memory list current.
    """
    return (
        db.query(Page)
        .filter(Page.document_id == doc_id)
        .order_by(Page.page_number)
        .all()
    )


def _stage_mark(
    doc: Document,
    stage_key: str,
    *,
    status: str,
    detail: str | None = None,
) -> None:
    """
    Record a compact stage result into doc.doc_metadata["stages"].

    Intentionally separate from job-tracking events:
        Job events                 "what happened when" trace, keyed by job_id.
        doc_metadata stage markers Compact per-document summary for quick
                                   post-mortems without joining the jobs table.

    The caller is responsible for committing after this call.
    """
    meta:   dict[str, Any] = dict(doc.doc_metadata or {})
    stages: dict[str, Any] = dict(meta.get("stages") or {})
    entry:  dict[str, Any] = {"status": status}
    if detail:
        entry["detail"] = detail
    stages[stage_key] = entry
    meta["stages"]    = stages
    doc.doc_metadata  = meta


# ─── Main worker entrypoint ───

def process_document(
    doc_id: int,
    mode: PipelineMode | str = PipelineMode.DEFAULT,
    stages: list[str] | None = None,
) -> None:
    """
    Execute the standard DB-backed pipeline for a single Document.

    Intended to be executed by an RQ worker; must not assume it is running in
    the FastAPI process.

    Args:
        doc_id:  Target document ID.
        mode:    Worker pipeline mode. Accepts PipelineMode or its canonical
                 string. The argument is accepted for queue/API compatibility;
                 per-mode stage semantics are not yet fully implemented.
        stages:  Optional future stage subset (e.g. `["s3_parse"]`). Accepted
                 for queue/API compatibility; stage selection is not yet implemented.

    On success: Document.status → completed, job status → finished.
    On fatal error (S0 or unexpected exception): Document.status → failed,
    job status → failed, exception re-raised so RQ marks the job failed.
    """
    mode = parse_pipeline_mode(mode)

    db: Session = get_sync_session()

    rq_job = get_current_job()
    job_id = rq_job.id if rq_job else f"doc-{doc_id}"

    logger.info(
        f"[OrchestratorStd] Starting pipeline doc_id={doc_id} "
        f"job_id={job_id} mode={mode.value} stages={stages}"
    )

    try:
        # ensure_job_row is inside the try block so db.close() in finally runs
        # even if job tracking raises unexpectedly.
        ensure_job_row(db, job_id=job_id, document_id=doc_id, stage="pipeline", status="started")
        append_event(db, job_id=job_id, level="info", stage="pipeline", message="Worker started")

        doc = _load_doc_or_fail(db, doc_id)

        doc.status = DocStatus.processing
        _stage_mark(doc, "pipeline", status="ok", detail="started")
        db.commit()
        append_event(db, job_id=job_id, level="info", stage="pipeline", message="Document -> processing")

        # ─── S0: Materialisation (mandatory) ───
        # Heavy imports are deferred into stage blocks so the web process never
        # loads ML/GPU modules at import time.
        append_event(db, job_id=job_id, level="info", stage="s0_materialize", message="Starting")
        try:
            from app.pipeline.s0_materialisation.materialise import materialise_document_pages

            pages = materialise_document_pages(db, doc)
            _stage_mark(doc, "s0_materialize", status="ok", detail=f"pages={len(pages)}")
            db.commit()
            append_event(
                db, job_id=job_id, level="info", stage="s0_materialize",
                message=f"Committed pages={len(pages)}",
            )

        except Exception as e:
            db.rollback()
            doc = db.get(Document, doc_id)
            _stage_mark(doc, "s0_materialize", status="failed", detail=str(e))
            doc.status = DocStatus.failed
            db.commit()
            append_event(
                db, job_id=job_id, level="error", stage="s0_materialize",
                message="Failed (fatal)", data={"error": str(e)},
            )
            set_status(db, job_id=job_id, status="failed")
            raise

        pages = _refresh_pages(db, doc.id)

        # ─── S1: Classification ───
        append_event(db, job_id=job_id, level="info", stage="s1_classify", message="Starting")
        try:
            input_kind = (doc.doc_metadata or {}).get("input_kind")

            if input_kind == "pdf":
                try:
                    from app.pipeline.s1_classification.classify_pdf import classify_pdf
                except Exception as imp_err:
                    _stage_mark(doc, "s1_classify", status="skipped", detail=f"import_error: {imp_err}")
                    db.commit()
                    append_event(
                        db, job_id=job_id, level="warning", stage="s1_classify",
                        message="Skipped (import_error)", data={"error": str(imp_err)},
                    )
                else:
                    classify_pdf(db, doc, pages, pdf_path=doc.source_uri, backend="auto")
                    _stage_mark(doc, "s1_classify", status="ok")
                    db.commit()
                    append_event(db, job_id=job_id, level="info", stage="s1_classify", message="Committed")

            else:
                for p in pages:
                    p.sourcetype = "image"
                doc.sourcetype = "image"
                _stage_mark(doc, "s1_classify", status="ok")
                db.commit()
                append_event(db, job_id=job_id, level="info", stage="s1_classify", message="Committed")

        except Exception as e:
            db.rollback()
            doc = db.get(Document, doc_id)
            _stage_mark(doc, "s1_classify", status="failed", detail=str(e))
            db.commit()
            logger.warning(f"[OrchestratorStd][S1] classify failed doc_id={doc_id}: {e}")
            append_event(
                db, job_id=job_id, level="warning", stage="s1_classify",
                message="Failed; continuing", data={"error": str(e)},
            )

        pages = _refresh_pages(db, doc.id)

        # ── S2: Detection (language + script) ─────────────────────────────────
        append_event(db, job_id=job_id, level="info", stage="s2_detection", message="Starting")
        try:
            try:
                from app.pipeline.s2_detection.lang_script_detect import detect_langs_and_scripts
            except Exception as imp_err:
                _stage_mark(doc, "s2_detection", status="skipped", detail=f"import_error: {imp_err}")
                db.commit()
                append_event(
                    db, job_id=job_id, level="warning", stage="s2_detection",
                    message="Skipped (import_error)", data={"error": str(imp_err)},
                )
            else:
                summary = detect_langs_and_scripts(db, doc=doc, pages=pages)
                _stage_mark(
                    doc, "s2_detection", status="ok",
                    detail=(
                        f"pages={summary.pages} "
                        f"pages_with_lang={summary.pages_with_lang} "
                        f"pages_with_script={summary.pages_with_script}"
                    ),
                )
                db.commit()
                append_event(
                    db, job_id=job_id, level="info", stage="s2_detection",
                    message=(
                        f"Committed doc_primary_lang={summary.doc_primary_lang} "
                        f"doc_primary_script={summary.doc_primary_script}"
                    ),
                )

        except Exception as e:
            db.rollback()
            doc = db.get(Document, doc_id)
            _stage_mark(doc, "s2_detection", status="failed", detail=str(e))
            db.commit()
            append_event(
                db, job_id=job_id, level="warning", stage="s2_detection",
                message="Failed; continuing", data={"error": str(e)},
            )

        pages = _refresh_pages(db, doc.id)

        # ─── S3: Parsing / DLA ───
        append_event(db, job_id=job_id, level="info", stage="s3_parse", message="Starting")
        try:
            from app.pipeline.s3_parsing.dla_stage import parse_and_persist_document
        except Exception as imp_err:
            _stage_mark(doc, "s3_parse", status="skipped", detail=f"import_error: {imp_err}")
            db.commit()
            append_event(
                db, job_id=job_id, level="warning", stage="s3_parse",
                message="Skipped (import_error)", data={"error": str(imp_err)},
            )
        else:
            try:
                summary = parse_and_persist_document(db, doc=doc, pages=pages, device="gpu", strict=False)
                _stage_mark(
                    doc, "s3_parse", status="ok",
                    detail=(
                        f"parsed_pages={summary.get('parsed_pages')} "
                        f"failed_pages={summary.get('failed_pages')}"
                    ),
                )
                db.commit()
                append_event(
                    db, job_id=job_id, level="info", stage="s3_parse",
                    message=(
                        f"Committed parsed_pages={summary.get('parsed_pages')} "
                        f"failed_pages={summary.get('failed_pages')}"
                    ),
                )
            except Exception as e:
                db.rollback()
                doc = db.get(Document, doc_id)
                _stage_mark(doc, "s3_parse", status="failed", detail=str(e))
                db.commit()
                logger.exception(f"[OrchestratorStd][S3] parse failed doc_id={doc_id}: {e}")
                append_event(
                    db, job_id=job_id, level="error", stage="s3_parse",
                    message="Failed; continuing", data={"error": str(e)},
                )

        pages = _refresh_pages(db, doc.id)

        # ── S4a: NLP — Machine Translation ────────────────────────────────────
        # Translates page.text_layer (populated by S3) into the target language.
        # Outputs: PageMT / BlockMT rows + per-page artifact pXXXX_text_mt.<lang>.md.
        # Skipped gracefully if mt_stage cannot be imported or all pages have no text_layer.
        append_event(db, job_id=job_id, level="info", stage="s4a_translate", message="Starting")
        try:
            try:
                from app.pipeline.s4_nlp.mt_stage import translate_and_persist_document
            except Exception as imp_err:
                _stage_mark(doc, "s4a_translate", status="skipped", detail=f"import_error: {imp_err}")
                db.commit()
                append_event(
                    db, job_id=job_id, level="warning", stage="s4a_translate",
                    message="Skipped (import_error)", data={"error": str(imp_err)},
                )
            else:
                if not any(p.text_layer for p in pages):
                    _stage_mark(
                        doc, "s4a_translate", status="skipped",
                        detail="no_text_layer — S3 may have failed or produced no text",
                    )
                    db.commit()
                    append_event(
                        db, job_id=job_id, level="warning", stage="s4a_translate",
                        message="Skipped (no text_layer on any page)",
                    )
                else:
                    mt_summary = translate_and_persist_document(db, doc=doc, pages=pages, strict=False)
                    _stage_mark(
                        doc, "s4a_translate", status="ok",
                        detail=(
                            f"translated_pages={mt_summary.get('translated_pages')} "
                            f"skipped_pages={mt_summary.get('skipped_pages')} "
                            f"failed_pages={mt_summary.get('failed_pages')} "
                            f"backend={mt_summary.get('backend')}"
                        ),
                    )
                    db.commit()
                    append_event(
                        db, job_id=job_id, level="info", stage="s4a_translate",
                        message=(
                            f"Committed translated_pages={mt_summary.get('translated_pages')} "
                            f"backend={mt_summary.get('backend')} model={mt_summary.get('model')}"
                        ),
                    )

        except Exception as e:
            db.rollback()
            doc = db.get(Document, doc_id)
            _stage_mark(doc, "s4a_translate", status="failed", detail=str(e))
            db.commit()
            logger.warning(f"[OrchestratorStd][S4a] translation failed doc_id={doc_id}: {e}")
            append_event(
                db, job_id=job_id, level="warning", stage="s4a_translate",
                message="Failed; continuing", data={"error": str(e)},
            )

        # pages = _refresh_pages(db, doc.id) # Unnecessary as S5 is not implemented

        # ─── S5: Embeddings (not yet implemented) ───

        # ─── Finalize ───
        doc = _load_doc_or_fail(db, doc_id)
        doc.status = DocStatus.completed
        _stage_mark(doc, "pipeline", status="ok", detail="completed")
        db.commit()

        append_event(db, job_id=job_id, level="info", stage="pipeline", message="Document -> completed")
        set_status(db, job_id=job_id, status="finished")
        logger.info(f"[OrchestratorStd] Completed pipeline doc_id={doc_id} job_id={job_id}")

    except Exception as e:
        # Fatal error path — S0 exceptions are handled explicitly above and
        # re-raised here. Also catches unexpected failures in orchestration logic.
        logger.exception(f"[OrchestratorStd] Pipeline failed doc_id={doc_id}: {e}")
        append_event(
            db, job_id=job_id, level="error", stage="pipeline",
            message="Pipeline exception", data={"error": str(e)},
        )
        try:
            doc = db.get(Document, doc_id)
            if doc is not None:
                doc.status = DocStatus.failed
                _stage_mark(doc, "pipeline", status="failed", detail=str(e))
                db.commit()
        except Exception as db_err:
            logger.error(
                f"[OrchestratorStd] Failed to update Document status after pipeline error: {db_err}"
            )
            append_event(
                db, job_id=job_id, level="error", stage="pipeline",
                message="Failed to update document status after exception",
                data={"db_error": str(db_err)},
            )
        set_status(db, job_id=job_id, status="failed")
        raise

    finally:
        db.close()






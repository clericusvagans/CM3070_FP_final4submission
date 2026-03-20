# app/pipeline/s3_parsing/paddle_singleton.py

"""
Worker-only Paddle parser singleton.

PaddleOCR / PPStructure models are expensive to load. In an RQ worker deployment, models should be loaded once per worker process and reused across jobs. At the same time, accidental model loads in the web process must be prevented: FastAPI imports routes at startup and any import chain that instantiates Paddle will trigger costly (and often broken) initialization in uvicorn.

Worker-only guard
-----------------
Model loading is permitted only when APP_PROCESS_ROLE=worker (set in workers.py).
For local debugging only: ALLOW_PADDLE_IN_WEB=1 bypasses the guard.

Language handling
-----------------
Language is page-specific and must not be pinned at singleton construction time. The singleton is keyed only by (pipeline, device, paddlex_config);
lang is intentionally excluded from singleton identity. Callers must pass per-page language into PaddleParser.parse(lang=...).
"""

import json
import os
from pathlib import Path
from threading import Lock
from typing import Any

from app.pipeline.s3_parsing.paddle_parse import PaddleParseConfig, PaddleParser
from app.utils.logger import logger

_LOCK: Lock = Lock()
_PARSER: Any | None = None
_CFG: PaddleParseConfig | None = None


# ─── Internal helpers ───

def _assert_worker_context() -> None:
    """
    Prevent model loading outside worker processes.
    Override for local debugging only: ALLOW_PADDLE_IN_WEB=1.
    Never enable the override in production web servers.
    """
    if os.getenv("ALLOW_PADDLE_IN_WEB", "0") != "1" and os.getenv("APP_PROCESS_ROLE", "") != "worker":
        raise RuntimeError(
            "Paddle models may only be loaded in worker processes. "
            "Set APP_PROCESS_ROLE=worker in the worker environment. "
            "For local debugging only, set ALLOW_PADDLE_IN_WEB=1."
        )


def _normalize_paddlex_config(paddlex_config: Any) -> Any:
    """
    Normalize paddlex_config so equality comparisons behave predictably.
    Supports None, dict (→ stable JSON string), Path/str (→ POSIX string).
    """
    if paddlex_config is None:
        return None
    if isinstance(paddlex_config, dict):
        return json.dumps(paddlex_config, sort_keys=True, separators=(",", ":"))
    if isinstance(paddlex_config, Path):
        return paddlex_config.as_posix()
    if isinstance(paddlex_config, str):
        try:
            return Path(paddlex_config).as_posix()
        except Exception:
            return paddlex_config
    return paddlex_config  # fallback; may break equality


def _normalized_cfg(cfg: PaddleParseConfig) -> PaddleParseConfig:
    """
    Return a config semantically equivalent to cfg but with normalized
    paddlex_config and lang stripped (lang is not part of singleton identity).
    """
    return PaddleParseConfig(
        pipeline=cfg.pipeline,
        device=cfg.device,
        lang=None,
        paddlex_config=_normalize_paddlex_config(cfg.paddlex_config),
    )


# ─── Public API ───

def reset_paddle_parser() -> None:
    """
    Reset the cached parser in the current process and aggressively release GPU memory.

    Safe to call in workers after S3 to ensure Paddle models do not remain
    resident in VRAM before an MT stage that also needs GPU headroom.

    Steps: drop the reference → call parser.close() → run gc.collect() →
    call paddle.device.cuda.empty_cache() (best-effort).
    """
    global _PARSER, _CFG

    with _LOCK:
        parser = _PARSER
        _PARSER = None
        _CFG = None

    try:
        if parser is not None and hasattr(parser, "close"):
            parser.close()
    except Exception:
        pass

    try:
        import gc
        gc.collect()
    except Exception:
        pass

    try:
        import paddle  # type: ignore
        if getattr(paddle, "is_compiled_with_cuda", lambda: False)():
            paddle.device.cuda.empty_cache()
    except Exception:
        pass

    logger.info("[PaddleSingleton] Parser cache reset (GPU cleanup attempted)")


def release_paddle_parser() -> None:
    """Alias for reset_paddle_parser(). Use at the end of S3 to reclaim VRAM."""
    reset_paddle_parser()


def get_paddle_parser(
    *,
    cfg: PaddleParseConfig | None = None,
    pipeline: str | None = None,
    device: str | None = None,
    lang: str | None = None,
    paddlex_config: Any | None = None,
    rebuild_on_change: bool = False,
) -> PaddleParser:
    """
    Return a process-wide (per-worker) singleton PaddleParser instance.

    The singleton is keyed by (pipeline, device, paddlex_config). `lang`
    is NOT part of the key — language must be passed per page into
    PaddleParser.parse(lang=...). This avoids freezing the worker to the
    first language seen while still reusing the heavy model state.

    Args:
        cfg:               Pre-built PaddleParseConfig (normalized before use).
        pipeline:          Pipeline name; used when cfg is None.
        device:            Device ("cpu" | "gpu"); used when cfg is None.
        lang:              Accepted for backward compatibility; ignored for keying.
        paddlex_config:    Optional PaddleX config; normalized into singleton identity.
        rebuild_on_change: If True, rebuild when a non-language config change is
                           detected. If False (default), raise RuntimeError instead.

    Raises:
        RuntimeError: If called outside a worker process (see _assert_worker_context),
                      or if config mismatch detected and rebuild_on_change is False.
    """
    global _PARSER, _CFG

    if cfg is None:
        cfg = PaddleParseConfig(
            pipeline=pipeline or "pp_structure_v3",
            device=device or "gpu",
            lang=None,
            paddlex_config=_normalize_paddlex_config(paddlex_config),
        )
    else:
        cfg = _normalized_cfg(cfg)

    with _LOCK:
        if _PARSER is None:
            logger.info(f"[PaddleSingleton] Initializing parser cfg={cfg}")
            _PARSER = PaddleParser(cfg)
            _CFG    = cfg
            return _PARSER

        if _CFG != cfg:
            msg = f"[PaddleSingleton] Existing cfg={_CFG} differs from requested cfg={cfg}"
            if rebuild_on_change:
                logger.warning(f"{msg}. Rebuilding because rebuild_on_change=True")
                _PARSER = PaddleParser(cfg)
                _CFG    = cfg
                return _PARSER
            raise RuntimeError(
                f"{msg}. Set rebuild_on_change=True to rebuild, or call reset_paddle_parser()."
            )

        return _PARSER
    


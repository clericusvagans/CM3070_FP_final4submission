# app/utils/logger.py
"""
Centralized logger.

All pipeline modules should import this logger instead of creating their own module-level handlers. This guarantees consistent formatting and prevents the same record from being emitted multiple times with different formats.

Note: Python loggers propagate to their parent loggers by default. If the root logger (or another ancestor) also has handlers attached, a single `logger.info(...)` call may appear twice:

1. once via the handler attached here
2. once again via the root / framework handler

To prevent this: `logger.propagate = False`.
"""

import logging
import sys

logger = logging.getLogger("pipeline")
logger.setLevel(logging.INFO)

# Prevent messages from bubbling up to the root logger and being emitted again
# by framework / RQ / uvicorn handlers.
logger.propagate = False

# Avoid adding multiple handlers when modules reimport.
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)s [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)


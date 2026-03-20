# app/workers.py

"""
RQ worker entry point for the AI orchestration pipeline.
"""

import os
from rq import Worker, Queue, SimpleWorker
from redis import Redis
from app.config.config import settings
from app.utils.logger import logger



def create_queues(conn: Redis) -> list[Queue]:
    """
    Currently a single queue drives the full sequential pipeline (S0-S5).
    For tuture development: I could decompose into per-stage queues to allow dedicated workers per hardware resource (e.g. dla_ocr and nlp on GPU, indexing on CPU).
    """
    return [
        Queue(settings.RQ_QUEUE_NAME, connection=conn),   # "pipeline" — active
        # Queue("dla_ocr", connection=conn),        # S3 Paddle — future
        # Queue("nlp", connection=conn),            # S4 MT     — future
        # Queue("indexing", connection=conn),       # S5        — future
        # Queue("low_priority", connection=conn),   # maintenance — future
    ]


def enforce_single_worker(conn: Redis) -> None:
    """
    Prevent multiple heavy workers from running simultaneously.
    Controlled by env flag RQ_SINGLE_WORKER=1.
    Uses SETNX semantics (set only if not exists).
    """
    if os.getenv("RQ_SINGLE_WORKER", "0") != "1":
        return

    key = "cm3070:lock:single_worker"
    acquired = conn.set(name=key, value=os.getpid(), nx=True)

    if not acquired:
        logger.error("Another worker already holds the single-worker lock. Exiting.")
        raise SystemExit(1)

    logger.info("Single-worker lock acquired.")


def main() -> None:
    os.environ.setdefault("APP_PROCESS_ROLE", "worker")

    conn = Redis.from_url(settings.REDIS_URL)
    enforce_single_worker(conn)
    queues = create_queues(conn)

    # NOTE: Here I use SimpleWorker (no fork) so the Paddle singleton (paddle_singleton.py)
    # persists across jobs. Worker forks a child per job, so the singleton would be
    # re-initialized on every job. SimpleWorker is acceptable here but not for production:
    # a crash (segfault, OOM in Paddle/CUDA) kills the entire worker process, not just
    # the failing job. Switch to Worker for production GPU workloads.
    worker = SimpleWorker(queues, connection=conn)

    worker.push_exc_handler(
        lambda job, exc_type, exc_value, traceback: logger.error(
            f"[Worker] Job {job.id} failed: {exc_value}"
        )
    )

    logger.info(f"[Worker] Ready: PID={os.getpid()}")
    worker.work(with_scheduler=True, burst=False)


if __name__ == "__main__":
    main()

    
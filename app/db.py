# app/db.py

"""
Database engine and session management for FastAPI (async) and RQ workers (sync).

IMPORTANT:
    FastAPI endpoints are async → must use AsyncSession.
    RQ workers run synchronous CPU/GPU tasks → must use sync Session.

Exposes:
    Base                → SQLAlchemy declarative base (shared by both engines)
    get_session()       → FastAPI async dependency
    get_sync_session()  → RQ worker / orchestrator helper (sync, caller must close)
    async_engine        → AsyncEngine for FastAPI
    sync_engine         → sync Engine for workers and Alembic

This separation avoids import errors, session misuse, uvicorn reload crashes,
and worker failures from AsyncSession misuse.
"""

from typing import AsyncGenerator
from sqlalchemy.orm import sessionmaker, declarative_base, Session
from sqlalchemy.ext.asyncio import create_async_engine, AsyncEngine, AsyncSession
from sqlalchemy import create_engine
from app.config.config import settings


Base = declarative_base()


# ──── Async engine (FastAPI) ────

async_engine: AsyncEngine = create_async_engine(
    settings.DB_ASYNC_URL,
    echo=False,
    future=True,
    pool_pre_ping=True,
)

AsyncSessionLocal = sessionmaker(
    bind=async_engine,
    expire_on_commit=False,
    class_=AsyncSession,
)


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI async dependency — yields a request-scoped AsyncSession."""
    async with AsyncSessionLocal() as session:
        yield session


# ──── Sync engine (RQ workers + Alembic) ────

sync_engine = create_engine(
    settings.DB_SYNC_URL,
    echo=False,
    future=True,
    pool_pre_ping=True,
)

SyncSessionLocal = sessionmaker(
    bind=sync_engine,
    autocommit=False,
    autoflush=False,
    expire_on_commit=False,
)


def get_sync_session() -> Session:
    """
    Return a new sync Session for use by RQ workers and the orchestrator.

    IMPORTANT: The caller is responsible for closing the session.
    """
    return SyncSessionLocal()
"""Shared asyncpg helpers for ingestion modules and scripts.

We use asyncpg directly (not SQLAlchemy) for ingestion because:
  - The schema is small and SQL is fine
  - asyncpg has the fastest Postgres driver in Python
  - No need for a sync driver (psycopg2) anywhere in the stack
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator

import asyncpg

from backend.config import get_settings


def asyncpg_dsn(database_url: str) -> str:
    """Strip the SQLAlchemy driver hint (postgresql+asyncpg://) for asyncpg."""
    return database_url.replace("postgresql+asyncpg://", "postgresql://", 1)


async def create_pool(min_size: int = 1, max_size: int = 5) -> asyncpg.Pool:
    """Create an asyncpg connection pool from settings.database_url.

    statement_cache_size=0 disables prepared statement caching, which is the
    safe default when connecting through Supabase's pgbouncer pooler. Has a
    minor performance cost but avoids the prepared-statement collision class
    of errors entirely.
    """
    settings = get_settings()
    if not settings.database_url:
        raise RuntimeError("DATABASE_URL is not set in .env")

    return await asyncpg.create_pool(
        asyncpg_dsn(settings.database_url),
        min_size=min_size,
        max_size=max_size,
        statement_cache_size=0,
        command_timeout=60,
    )


@asynccontextmanager
async def pool_context(**kwargs) -> AsyncIterator[asyncpg.Pool]:
    """`async with pool_context() as pool: ...` — closes the pool on exit."""
    pool = await create_pool(**kwargs)
    try:
        yield pool
    finally:
        await pool.close()

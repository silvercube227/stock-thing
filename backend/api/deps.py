"""Shared FastAPI dependencies.

Kept separate from main.py so routers can import `get_pool` without a circular
import (main imports the routers).
"""

from __future__ import annotations

import asyncpg
from fastapi import Request


def get_pool(request: Request) -> asyncpg.Pool:
    """Return the shared asyncpg pool created in the app lifespan."""
    return request.app.state.pool

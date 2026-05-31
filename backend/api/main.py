"""FastAPI app for the dashboard.

Owns a single asyncpg pool (created on startup, closed on shutdown) shared by all
routers via `backend.api.deps.get_pool`. CORS is open to the local Next.js dev
origin. No ML imports here — predictions are read from the DB, quotes from
yfinance — so the libomp torch/lightgbm segfault class is avoided entirely.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api.quotes import router as quotes_router
from backend.api.routers.portfolio import router as portfolio_router
from backend.api.routers.rankings import router as rankings_router
from backend.api.routers.tickers import router as tickers_router
from backend.config import get_settings
from backend.ingestion.db import create_pool

_extra_origins = [o.strip() for o in get_settings().cors_origins.split(",") if o.strip()]
ALLOWED_ORIGINS = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    *_extra_origins,
]


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.pool = await create_pool(min_size=1, max_size=5)
    try:
        yield
    finally:
        await app.state.pool.close()


app = FastAPI(title="stock-thing", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(portfolio_router)
app.include_router(tickers_router)
app.include_router(rankings_router)
app.include_router(quotes_router)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "backend.api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True,
    )

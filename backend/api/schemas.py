"""Pydantic request/response models for the dashboard API.

Field names follow the *rank* semantics of the production GBDT ranker:
`predictions.direction_prob` is a clipped percentile rank in [0, 1], not a
calibrated probability, so we surface it as `percentile_rank`. `predicted_return`
is null for ranker rows and is intentionally not exposed.
"""

from __future__ import annotations

from datetime import date

from pydantic import BaseModel, Field

# =============================================================
# Portfolio
# =============================================================


class HoldingCreate(BaseModel):
    """Body for POST /portfolio — add or update a holding by symbol."""

    symbol: str
    shares: float = Field(ge=0)
    cost_basis: float | None = None
    acquired_at: date | None = None
    notes: str | None = None


class HoldingUpdate(BaseModel):
    """Body for PATCH /portfolio/{ticker_id}."""

    shares: float | None = Field(default=None, ge=0)
    cost_basis: float | None = None


class PortfolioRow(BaseModel):
    ticker_id: int
    symbol: str
    name: str | None = None
    sector: str | None = None
    shares: float
    cost_basis: float | None = None
    acquired_at: date | None = None
    # Latest stored close — the client layers live quotes on top for net value.
    last_close: float | None = None
    last_close_date: date | None = None


# =============================================================
# Quotes
# =============================================================


class Quote(BaseModel):
    symbol: str
    price: float | None = None
    prev_close: float | None = None
    change: float | None = None
    change_pct: float | None = None
    # True when the value came from the stored daily close, not a live fetch.
    stale: bool = False


# =============================================================
# Ticker detail
# =============================================================


class HorizonPrediction(BaseModel):
    horizon: str
    # Clipped percentile rank in [0, 1] vs the active cross-section.
    percentile_rank: float
    # Rank stability: std of the predicted rank over the last up-to-3 scoring
    # dates (lower = steadier). Null when there isn't enough history yet.
    rank_std: float | None = None


class FundamentalsSnapshot(BaseModel):
    period_end: date | None = None
    filed_at: date | None = None
    filing_type: str | None = None
    revenue: float | None = None
    net_income: float | None = None
    gross_margin: float | None = None
    operating_margin: float | None = None
    total_debt: float | None = None
    total_equity: float | None = None
    fcf: float | None = None


class SentimentSnapshot(BaseModel):
    score_date: date | None = None
    mean_score: float | None = None
    headline_count: int | None = None
    rolling_7d: float | None = None
    rolling_14d: float | None = None


class TickerSummary(BaseModel):
    ticker_id: int
    symbol: str
    name: str | None = None
    sector: str | None = None
    industry: str | None = None
    asset_type: str | None = None


class TickerDetail(BaseModel):
    ticker: TickerSummary
    as_of_date: date | None = None
    model_version_id: str | None = None
    model_status: str | None = None
    predictions: list[HorizonPrediction] = Field(default_factory=list)
    fundamentals: FundamentalsSnapshot | None = None
    sentiment: SentimentSnapshot | None = None
    last_close: float | None = None


class PricePoint(BaseModel):
    date: date
    close: float


# =============================================================
# Cross-sectional rankings (screener)
# =============================================================


class RankingRow(BaseModel):
    ticker_id: int
    symbol: str
    name: str | None = None
    sector: str | None = None
    percentile_rank: float
    rank_std: float | None = None


class RankingResponse(BaseModel):
    horizon: str
    as_of_date: date | None = None
    model_version_id: str | None = None
    model_status: str | None = None
    rows: list[RankingRow] = Field(default_factory=list)

"""API tests with the DB pool and auth dependencies overridden.

We don't stand up Postgres here; instead a small in-memory `FakePool` answers the
exact queries the routers issue (dispatched by SQL substring). This exercises the
routing, user scoping, request/response shapes, and the auth boundary.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

import pytest
from fastapi.testclient import TestClient

from backend.api import quotes as quotes_mod
from backend.api.auth import get_current_user
from backend.api.deps import get_pool
from backend.api.main import app

USER = "user-1"


# =============================================================
# In-memory fake pool
# =============================================================


class _Store:
    def __init__(self) -> None:
        self.catalog = {
            "AAPL": {
                "ticker_id": 1, "symbol": "AAPL", "name": "Apple Inc.",
                "sector": "Technology", "industry": "Consumer Electronics",
                "asset_type": "equity",
            },
            "MSFT": {
                "ticker_id": 2, "symbol": "MSFT", "name": "Microsoft",
                "sector": "Technology", "industry": "Software",
                "asset_type": "equity",
            },
        }
        self.closes = {1: (190.0, "2026-05-22"), 2: (50.0, "2026-05-22")}
        self.holdings: dict[tuple[str, int], dict] = {}
        self.active_model: tuple[str, str] | None = None
        self.predictions: list[dict] = []
        self.rankings: list[dict] = []
        self.fundamentals_row: dict | None = None
        self.sentiment_row: dict | None = None

    def _by_id(self, ticker_id: int) -> dict | None:
        for v in self.catalog.values():
            if v["ticker_id"] == ticker_id:
                return v
        return None

    def _portfolio_row(self, user_id: str, ticker_id: int) -> dict | None:
        h = self.holdings.get((user_id, ticker_id))
        if h is None:
            return None
        cat = self._by_id(ticker_id) or {}
        close, cdate = self.closes.get(ticker_id, (None, None))
        return {
            "ticker_id": ticker_id, "symbol": cat.get("symbol"),
            "name": cat.get("name"), "sector": cat.get("sector"),
            "shares": h["shares"], "cost_basis": h.get("cost_basis"),
            "acquired_at": h.get("acquired_at"),
            "last_close": close, "last_close_date": cdate,
        }


class _Dispatcher:
    def __init__(self, store: _Store) -> None:
        self.store = store

    async def fetchrow(self, sql: str, *args):
        s = self.store
        sl = sql.lower()
        if "from portfolio_holdings h" in sl and "h.ticker_id = $2" in sl:
            return s._portfolio_row(args[0], args[1])
        if "from tickers where upper(symbol)" in sl and "active = true" in sl:
            cat = s.catalog.get(str(args[0]).upper())
            return {"ticker_id": cat["ticker_id"]} if cat else None
        if "from tickers where upper(symbol)" in sl:
            return s.catalog.get(str(args[0]).upper())
        if "select id from portfolio_holdings where user_id" in sl:
            return {"id": "x"} if (args[0], args[1]) in s.holdings else None
        if "from model_versions" in sl and "status = 'production'" in sl:
            if s.active_model and s.active_model[1] == "production":
                return {"model_version_id": s.active_model[0], "status": "production"}
            return None
        if "from model_versions" in sl and "order by created_at desc" in sl:
            if s.active_model:
                return {"model_version_id": s.active_model[0], "status": s.active_model[1]}
            return None
        if "from fundamentals where ticker_id" in sl:
            return s.fundamentals_row
        if "from sentiment_daily where ticker_id" in sl:
            return s.sentiment_row
        if "from price_history where ticker_id = $1 order by trade_date desc" in sl:
            close, _ = s.closes.get(args[0], (None, None))
            return {"close": close}
        raise AssertionError(f"unexpected fetchrow: {sql}")

    async def fetch(self, sql: str, *args):
        s = self.store
        sl = sql.lower()
        if "from portfolio_holdings h" in sl and "order by t.symbol" in sl:
            rows = [s._portfolio_row(u, t) for (u, t) in s.holdings if u == args[0]]
            return [r for r in rows if r]
        if "from tickers" in sl and "ilike" in sl:
            return list(s.catalog.values())
        if "from predictions p" in sl and "join tickers" in sl:
            return s.rankings
        if "from predictions" in sl:
            return s.predictions
        if "select t.symbol" in sl and "from tickers t" in sl:
            want = {x.upper() for x in args[0]}
            out = []
            for sym, cat in s.catalog.items():
                if sym in want:
                    close, _ = s.closes.get(cat["ticker_id"], (None, None))
                    out.append({"symbol": cat["symbol"], "close": close})
            return out
        if "from price_history" in sl and "adj_close" in sl:
            return []
        raise AssertionError(f"unexpected fetch: {sql}")

    async def execute(self, sql: str, *args) -> str:
        s = self.store
        sl = sql.lower()
        if "insert into portfolio_holdings" in sl:
            s.holdings[(args[0], args[1])] = {
                "shares": args[2], "cost_basis": args[3],
                "acquired_at": args[4], "notes": args[5],
            }
            return "INSERT 0 1"
        if "update portfolio_holdings" in sl and "coalesce" in sl:
            key = (args[0], args[1])
            if key not in s.holdings:
                return "UPDATE 0"
            if args[2] is not None:
                s.holdings[key]["shares"] = args[2]
            if args[3] is not None:
                s.holdings[key]["cost_basis"] = args[3]
            return "UPDATE 1"
        if "update portfolio_holdings" in sl:
            key = (args[0], args[1])
            if key not in s.holdings:
                return "UPDATE 0"
            s.holdings[key].update(
                {"shares": args[2], "cost_basis": args[3],
                 "acquired_at": args[4], "notes": args[5]}
            )
            return "UPDATE 1"
        if "delete from portfolio_holdings" in sl:
            return "DELETE 1" if s.holdings.pop((args[0], args[1]), None) else "DELETE 0"
        raise AssertionError(f"unexpected execute: {sql}")


class _FakeConn(_Dispatcher):
    @asynccontextmanager
    async def transaction(self):
        yield


class FakePool(_Dispatcher):
    @asynccontextmanager
    async def acquire(self):
        yield _FakeConn(self.store)


# =============================================================
# Fixtures
# =============================================================


@pytest.fixture
def store() -> _Store:
    return _Store()


@pytest.fixture
def client(store: _Store):
    # No context manager => the real lifespan (which connects to Postgres) does not
    # run; get_pool is overridden with the in-memory fake instead.
    pool = FakePool(store)
    app.dependency_overrides[get_pool] = lambda: pool
    app.dependency_overrides[get_current_user] = lambda: USER
    quotes_mod._cache.clear()
    yield TestClient(app)
    app.dependency_overrides.clear()


@pytest.fixture
def anon_client(store: _Store):
    """Client with auth NOT overridden, to test the 401 boundary."""
    app.dependency_overrides[get_pool] = lambda: FakePool(store)
    yield TestClient(app)
    app.dependency_overrides.clear()


# =============================================================
# Tests
# =============================================================


def test_health(client) -> None:
    assert client.get("/health").json() == {"status": "ok"}


def test_portfolio_requires_auth(anon_client) -> None:
    assert anon_client.get("/portfolio").status_code == 401


def test_quotes_requires_auth(anon_client) -> None:
    assert anon_client.get("/quotes?symbols=AAPL").status_code == 401


def test_portfolio_crud_roundtrip(client, store) -> None:
    # Add
    r = client.post("/portfolio", json={"symbol": "AAPL", "shares": 10})
    assert r.status_code == 201
    body = r.json()
    assert body["symbol"] == "AAPL" and body["shares"] == 10 and body["last_close"] == 190.0

    # It is scoped to USER
    assert (USER, 1) in store.holdings

    # List
    rows = client.get("/portfolio").json()
    assert len(rows) == 1 and rows[0]["ticker_id"] == 1

    # Patch shares
    r = client.patch("/portfolio/1", json={"shares": 25})
    assert r.status_code == 200 and r.json()["shares"] == 25

    # Delete
    assert client.delete("/portfolio/1").status_code == 204
    assert client.get("/portfolio").json() == []


def test_add_unknown_symbol_404(client) -> None:
    assert client.post("/portfolio", json={"symbol": "NOPE", "shares": 1}).status_code == 404


def test_patch_missing_holding_404(client) -> None:
    assert client.patch("/portfolio/1", json={"shares": 5}).status_code == 404


def test_ticker_search(client) -> None:
    rows = client.get("/tickers?q=app").json()
    assert any(r["symbol"] == "AAPL" for r in rows)


def test_ticker_detail_no_model(client) -> None:
    # No model registered -> empty predictions, but ticker info + close still returned.
    body = client.get("/tickers/AAPL").json()
    assert body["ticker"]["symbol"] == "AAPL"
    assert body["predictions"] == []
    assert body["last_close"] == 190.0


def test_ticker_detail_with_predictions(client, store) -> None:
    store.active_model = ("mv-123", "candidate")
    store.predictions = [
        {"horizon": "3M", "direction_prob": 0.82, "confidence": 0.64, "as_of_date": "2026-05-22"},
        {"horizon": "6M", "direction_prob": 0.55, "confidence": 0.10, "as_of_date": "2026-05-22"},
    ]
    body = client.get("/tickers/AAPL").json()
    assert body["model_status"] == "candidate"
    assert body["model_version_id"] == "mv-123"
    horizons = {p["horizon"]: p["percentile_rank"] for p in body["predictions"]}
    assert horizons == {"3M": 0.82, "6M": 0.55}


def test_rankings_requires_auth(anon_client) -> None:
    assert anon_client.get("/rankings?horizon=6M").status_code == 401


def test_rankings(client, store) -> None:
    store.active_model = ("mv-123", "production")
    store.rankings = [
        {"ticker_id": 2, "symbol": "MSFT", "name": "Microsoft", "sector": "Technology",
         "direction_prob": 0.91, "confidence": 0.82, "as_of_date": "2026-05-22"},
        {"ticker_id": 1, "symbol": "AAPL", "name": "Apple Inc.", "sector": "Technology",
         "direction_prob": 0.40, "confidence": 0.20, "as_of_date": "2026-05-22"},
    ]
    body = client.get("/rankings?horizon=6M").json()
    assert body["horizon"] == "6M"
    assert body["model_status"] == "production"
    assert body["as_of_date"] == "2026-05-22"
    syms = [r["symbol"] for r in body["rows"]]
    assert syms == ["MSFT", "AAPL"]  # order preserved from the query
    assert body["rows"][0]["percentile_rank"] == 0.91
    # Within-sector rank: both are Technology, MSFT (higher score) tops the sector.
    msft, aapl = body["rows"][0], body["rows"][1]
    assert msft["sector_rank"] == 1.0 and msft["sector_rank_label"] == "1/2"
    assert aapl["sector_rank"] == 0.0 and aapl["sector_rank_label"] == "2/2"
    # No price history in the fake store ⇒ Sharpe is null, not an error.
    assert msft["sharpe"] is None


def test_quotes_live(client, monkeypatch) -> None:
    monkeypatch.setattr(quotes_mod, "_fetch_one_sync", lambda sym: (100.0, 90.0))
    body = client.get("/quotes?symbols=AAPL").json()
    q = body["AAPL"]
    assert q["price"] == 100.0 and q["prev_close"] == 90.0
    assert q["change"] == 10.0 and abs(q["change_pct"] - 11.111) < 0.01
    assert q["stale"] is False


def test_quotes_fallback_to_stored_close(client, monkeypatch) -> None:
    monkeypatch.setattr(quotes_mod, "_fetch_one_sync", lambda sym: (None, None))
    body = client.get("/quotes?symbols=MSFT").json()
    q = body["MSFT"]
    assert q["price"] == 50.0 and q["stale"] is True

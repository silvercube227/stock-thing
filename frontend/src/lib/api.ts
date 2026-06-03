import { supabase } from "./supabase";

const BASE = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";

// ---- Types (mirror backend/api/schemas.py) ----

export interface PortfolioRow {
  ticker_id: number;
  symbol: string;
  name: string | null;
  sector: string | null;
  shares: number;
  cost_basis: number | null;
  acquired_at: string | null;
  last_close: number | null;
  last_close_date: string | null;
}

export interface Quote {
  symbol: string;
  price: number | null;
  prev_close: number | null;
  change: number | null;
  change_pct: number | null;
  stale: boolean;
}

export interface TickerSummary {
  ticker_id: number;
  symbol: string;
  name: string | null;
  sector: string | null;
  industry: string | null;
  asset_type: string | null;
  // Off-index ticker the user added: scored but not trained on. Drives the
  // accuracy disclaimer.
  user_added: boolean;
}

export interface AddTickerResponse {
  symbol: string;
  status: string; // "queued" | "exists"
  run_id: number | null;
  ticker_id: number | null;
}

export interface TickerStatus {
  symbol: string;
  // "running" | "ready" | "insufficient_history" | "failed" | "unknown"
  status: string;
  message: string | null;
}

export interface HorizonPrediction {
  horizon: string;
  percentile_rank: number;
  // Std of the predicted rank over the last up-to-3 scoring dates (lower =
  // steadier). Null until there's enough scoring history.
  rank_std: number | null;
}

export interface FundamentalsSnapshot {
  period_end: string | null;
  filed_at: string | null;
  filing_type: string | null;
  revenue: number | null;
  net_income: number | null;
  gross_margin: number | null;
  operating_margin: number | null;
  total_debt: number | null;
  total_equity: number | null;
  fcf: number | null;
}

export interface SentimentSnapshot {
  score_date: string | null;
  mean_score: number | null;
  headline_count: number | null;
  rolling_7d: number | null;
  rolling_14d: number | null;
}

export interface ValuationSnapshot {
  symbol: string;
  trailing_pe: number | null;
  forward_pe: number | null;
  price_to_sales: number | null;
  ebitda: number | null;
  // Fundamentals fallback for off-index names with no EDGAR filing.
  revenue: number | null;
  net_income: number | null;
  gross_margin: number | null;
  operating_margin: number | null;
  fcf: number | null;
}

export interface TickerDetail {
  ticker: TickerSummary;
  as_of_date: string | null;
  model_version_id: string | null;
  model_status: string | null;
  predictions: HorizonPrediction[];
  fundamentals: FundamentalsSnapshot | null;
  sentiment: SentimentSnapshot | null;
  last_close: number | null;
}

export interface PricePoint {
  date: string;
  close: number;
  // OHLC (split-adjusted, same axis as close) + volume for candlestick view.
  open: number | null;
  high: number | null;
  low: number | null;
  volume: number | null;
}

export interface RankingRow {
  ticker_id: number;
  symbol: string;
  name: string | null;
  sector: string | null;
  percentile_rank: number;
  rank_std: number | null;
  // Within-sector percentile in [0, 1] (null when sector unknown or <2 names).
  sector_rank: number | null;
  // Ordinal position within sector, e.g. "3/42".
  sector_rank_label: string | null;
  // Trailing realized annualized Sharpe (backward-looking, not a forecast).
  sharpe: number | null;
}

export interface RankingResponse {
  horizon: string;
  as_of_date: string | null;
  model_version_id: string | null;
  model_status: string | null;
  rows: RankingRow[];
}

// ---- Fetch wrapper ----

async function authHeader(): Promise<Record<string, string>> {
  const { data } = await supabase.auth.getSession();
  const token = data.session?.access_token;
  return token ? { Authorization: `Bearer ${token}` } : {};
}

async function apiFetch<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...(await authHeader()),
      ...(init?.headers ?? {}),
    },
  });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = body.detail ?? detail;
    } catch {
      /* non-JSON error body */
    }
    throw new Error(`${res.status}: ${detail}`);
  }
  if (res.status === 204) return undefined as T;
  return (await res.json()) as T;
}

// ---- Endpoints ----

export const getPortfolio = () => apiFetch<PortfolioRow[]>("/portfolio");

export const upsertHolding = (
  symbol: string,
  shares: number,
  cost_basis?: number | null,
) =>
  apiFetch<PortfolioRow>("/portfolio", {
    method: "POST",
    body: JSON.stringify({ symbol, shares, cost_basis }),
  });

export const patchShares = (ticker_id: number, shares: number) =>
  apiFetch<PortfolioRow>(`/portfolio/${ticker_id}`, {
    method: "PATCH",
    body: JSON.stringify({ shares }),
  });

export const deleteHolding = (ticker_id: number) =>
  apiFetch<void>(`/portfolio/${ticker_id}`, { method: "DELETE" });

export const getQuotes = (symbols: string[]) =>
  symbols.length
    ? apiFetch<Record<string, Quote>>(
        `/quotes?symbols=${encodeURIComponent(symbols.join(","))}`,
      )
    : Promise.resolve<Record<string, Quote>>({});

export const searchTickers = (q: string) =>
  apiFetch<TickerSummary[]>(`/tickers?q=${encodeURIComponent(q)}`);

export const getTickerDetail = (symbol: string) =>
  apiFetch<TickerDetail>(`/tickers/${encodeURIComponent(symbol)}`);

export const getPrices = (symbol: string, lookback = "1y") =>
  apiFetch<PricePoint[]>(
    `/tickers/${encodeURIComponent(symbol)}/prices?lookback=${lookback}`,
  );

export const getValuation = (symbol: string) =>
  apiFetch<ValuationSnapshot>(
    `/tickers/${encodeURIComponent(symbol)}/valuation`,
  );

export const getRankings = (horizon: string, limit = 500) =>
  apiFetch<RankingResponse>(`/rankings?horizon=${horizon}&limit=${limit}`);

export const addTicker = (symbol: string) =>
  apiFetch<AddTickerResponse>("/tickers", {
    method: "POST",
    body: JSON.stringify({ symbol }),
  });

export const getTickerStatus = (symbol: string) =>
  apiFetch<TickerStatus>(`/tickers/${encodeURIComponent(symbol)}/status`);

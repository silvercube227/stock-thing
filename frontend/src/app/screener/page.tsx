"use client";

import { useEffect, useMemo, useState } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { useAuth } from "@/components/AuthProvider";
import { AppHeader } from "@/components/AppHeader";
import { getRankings, type RankingResponse, type RankingRow } from "@/lib/api";
import { percentileRank, num, changeColor } from "@/lib/format";
import { usePortfolio } from "@/hooks/usePortfolio";

const HORIZONS = ["3M", "6M", "1Y"];

type SortKey = "rank" | "sharpe";

function barTone(rank: number): string {
  if (rank >= 0.66) return "bg-up";
  if (rank <= 0.34) return "bg-down";
  return "bg-accent";
}

export default function ScreenerPage() {
  const router = useRouter();
  const { session, loading: authLoading } = useAuth();
  const { rows: holdings } = usePortfolio();

  const [horizon, setHorizon] = useState("6M");
  const [data, setData] = useState<RankingResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [filter, setFilter] = useState("");
  const [heldOnly, setHeldOnly] = useState(false);
  const [sectorMode, setSectorMode] = useState(false);
  const [sortKey, setSortKey] = useState<SortKey>("rank");
  const [sectorFilter, setSectorFilter] = useState("");
  const [minSharpe, setMinSharpe] = useState("");

  useEffect(() => {
    if (!authLoading && !session) router.replace("/login");
  }, [authLoading, session, router]);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);
    getRankings(horizon)
      .then((d) => !cancelled && setData(d))
      .catch((e) => !cancelled && setError(e instanceof Error ? e.message : "Failed"))
      .finally(() => !cancelled && setLoading(false));
    return () => {
      cancelled = true;
    };
  }, [horizon]);

  const heldIds = useMemo(
    () => new Set(holdings.map((h) => h.ticker_id)),
    [holdings],
  );

  // Lens-aware rank value for a row: within-sector percentile when the sector
  // toggle is on (null if the name has no sector rank), else the universe rank.
  const lensRank = (r: RankingRow): number | null =>
    sectorMode ? r.sector_rank : r.percentile_rank;

  const sectors = useMemo(() => {
    const s = new Set<string>();
    for (const r of data?.rows ?? []) if (r.sector) s.add(r.sector);
    return [...s].sort();
  }, [data]);

  const visible = useMemo(() => {
    const all = data?.rows ?? [];
    const q = filter.trim().toLowerCase();
    const minS = minSharpe.trim() === "" ? null : Number(minSharpe);
    const filtered = all.filter((r) => {
      if (heldOnly && !heldIds.has(r.ticker_id)) return false;
      if (sectorFilter && r.sector !== sectorFilter) return false;
      if (minS != null && !Number.isNaN(minS) && (r.sharpe == null || r.sharpe < minS))
        return false;
      if (!q) return true;
      return (
        r.symbol.toLowerCase().includes(q) ||
        (r.name ?? "").toLowerCase().includes(q) ||
        (r.sector ?? "").toLowerCase().includes(q)
      );
    });
    // Sort descending (higher = better); nulls sink to the bottom either way.
    const keyOf = (r: RankingRow): number | null =>
      sortKey === "sharpe" ? r.sharpe : lensRank(r);
    return [...filtered].sort((a, b) => {
      const ka = keyOf(a);
      const kb = keyOf(b);
      if (ka == null && kb == null) return 0;
      if (ka == null) return 1;
      if (kb == null) return -1;
      return kb - ka;
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [data, filter, heldOnly, heldIds, sectorMode, sortKey, sectorFilter, minSharpe]);

  if (authLoading || !session) {
    return (
      <main className="flex flex-1 items-center justify-center text-sm text-muted">
        Loading…
      </main>
    );
  }

  return (
    <>
      <AppHeader />
      <main className="mx-auto w-full max-w-5xl flex-1 px-6 py-8">
        <div className="mb-6 flex flex-wrap items-end justify-between gap-4">
          <div>
            <h1 className="text-xl font-semibold tracking-tight">Screener</h1>
            <p className="mt-1 text-sm text-muted">
              {sectorMode
                ? "Ranked within each sector — how every name stacks up against its own peers."
                : "Universe ranked by relative strength."}
              {data?.as_of_date ? ` As of ${data.as_of_date}.` : ""}
            </p>
          </div>
          <div className="flex rounded-xl border border-border bg-surface p-1 text-sm">
            {HORIZONS.map((h) => (
              <button
                key={h}
                onClick={() => setHorizon(h)}
                className={`rounded-lg px-4 py-1.5 text-xs font-medium transition-colors ${
                  horizon === h
                    ? "bg-accent/15 text-accent"
                    : "text-muted hover:text-foreground"
                }`}
              >
                {h}
              </button>
            ))}
          </div>
        </div>

        <div className="mb-4 flex flex-wrap items-center gap-3">
          <input
            value={filter}
            onChange={(e) => setFilter(e.target.value)}
            placeholder="Filter by symbol, name, or sector…"
            className="flex-1 rounded-xl border border-border bg-surface px-4 py-2.5 text-sm outline-none transition-colors placeholder:text-faint focus:border-accent/60"
          />
          <select
            value={sectorFilter}
            onChange={(e) => setSectorFilter(e.target.value)}
            className="rounded-xl border border-border bg-surface px-3 py-2.5 text-sm text-muted outline-none transition-colors focus:border-accent/60"
          >
            <option value="">All sectors</option>
            {sectors.map((s) => (
              <option key={s} value={s}>
                {s}
              </option>
            ))}
          </select>
          <input
            value={minSharpe}
            onChange={(e) => setMinSharpe(e.target.value)}
            inputMode="decimal"
            placeholder="Min Sharpe"
            className="w-28 rounded-xl border border-border bg-surface px-3 py-2.5 text-sm outline-none transition-colors placeholder:text-faint focus:border-accent/60"
          />
          <label className="flex items-center gap-2 text-xs text-muted cursor-pointer">
            <input
              type="checkbox"
              checked={sectorMode}
              onChange={(e) => setSectorMode(e.target.checked)}
              className="accent-accent"
            />
            Rank within sector
          </label>
          <label className="flex items-center gap-2 text-xs text-muted cursor-pointer">
            <input
              type="checkbox"
              checked={heldOnly}
              onChange={(e) => setHeldOnly(e.target.checked)}
              className="accent-accent"
            />
            My holdings only
          </label>
        </div>

        {error && (
          <p className="rounded-xl border border-down/30 bg-down/8 px-4 py-2.5 text-sm text-down mb-4">
            {error}
          </p>
        )}

        {loading ? (
          <div className="rounded-2xl border border-border p-12 text-center text-sm text-muted">
            Loading rankings…
          </div>
        ) : visible.length === 0 ? (
          <div className="rounded-2xl border border-dashed border-border p-12 text-center text-sm text-muted">
            No tickers match.
          </div>
        ) : (
          <div className="overflow-hidden rounded-2xl border border-border">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-border bg-surface-2 text-left text-xs text-muted">
                  <th className="w-14 px-5 py-3.5 text-right font-medium">#</th>
                  <th className="px-5 py-3.5 font-medium">Ticker</th>
                  <th className="px-5 py-3.5 font-medium">
                    {sectorMode ? "Within-sector strength" : "Relative strength"}
                  </th>
                  <th
                    onClick={() => setSortKey("rank")}
                    className={`cursor-pointer px-5 py-3.5 text-right font-medium select-none hover:text-foreground ${
                      sortKey === "rank" ? "text-accent" : ""
                    }`}
                  >
                    {sectorMode ? "Sector pct" : "Percentile"}
                    {sortKey === "rank" ? " ↓" : ""}
                  </th>
                  <th
                    onClick={() => setSortKey("sharpe")}
                    title="Trailing 1-year realized annualized Sharpe — backward-looking, not a forecast."
                    className={`cursor-pointer px-5 py-3.5 text-right font-medium select-none hover:text-foreground ${
                      sortKey === "sharpe" ? "text-accent" : ""
                    }`}
                  >
                    Sharpe (1Y)
                    {sortKey === "sharpe" ? " ↓" : ""}
                  </th>
                </tr>
              </thead>
              <tbody>
                {visible.map((r, i) => {
                  const rank = lensRank(r);
                  const pctile = percentileRank(rank);
                  const held = heldIds.has(r.ticker_id);
                  const posLabel = sectorMode
                    ? (r.sector_rank_label ?? "—")
                    : i + 1;
                  return (
                    <tr
                      key={r.ticker_id}
                      onClick={() => router.push(`/ticker/${r.symbol}`)}
                      className={`cursor-pointer border-b border-border/40 last:border-0 transition-colors hover:bg-surface ${
                        held ? "bg-accent/4" : ""
                      }`}
                    >
                      <td className="nums px-5 py-4 text-right text-faint text-xs">
                        {posLabel}
                      </td>
                      <td className="px-5 py-4">
                        <div className="flex items-center gap-2">
                          <span className="font-semibold tracking-wide">{r.symbol}</span>
                          {held && (
                            <span className="rounded-sm bg-accent/15 px-1.5 py-0.5 text-[9px] uppercase tracking-widest text-accent">
                              held
                            </span>
                          )}
                        </div>
                        <div className="max-w-[16rem] truncate text-[11px] text-muted mt-0.5">
                          {sectorMode
                            ? (r.sector ?? r.name ?? "")
                            : (r.name ?? r.sector ?? "")}
                        </div>
                      </td>
                      <td className="px-5 py-4">
                        <div className="h-1.5 w-full max-w-[14rem] overflow-hidden rounded-full bg-surface-2">
                          {rank != null && (
                            <div
                              className={`h-full rounded-full ${barTone(rank)}`}
                              style={{ width: `${rank * 100}%` }}
                            />
                          )}
                        </div>
                      </td>
                      <td className="nums px-5 py-4 text-right font-medium">{pctile}</td>
                      <td
                        className={`nums px-5 py-4 text-right font-medium ${changeColor(r.sharpe)}`}
                      >
                        {num(r.sharpe, 2)}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}

        <div className="mt-6">
          <Link href="/" className="text-[11px] text-faint transition-colors hover:text-foreground">
            ← Back to portfolio
          </Link>
        </div>
      </main>
    </>
  );
}

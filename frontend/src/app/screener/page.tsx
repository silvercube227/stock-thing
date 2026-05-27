"use client";

import { useEffect, useMemo, useState } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { useAuth } from "@/components/AuthProvider";
import { AppHeader } from "@/components/AppHeader";
import { getRankings, type RankingResponse } from "@/lib/api";
import { usePortfolio } from "@/hooks/usePortfolio";

const HORIZONS = ["3M", "6M", "1Y"];

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

  const visible = useMemo(() => {
    const all = data?.rows ?? [];
    const q = filter.trim().toLowerCase();
    return all.filter((r) => {
      if (heldOnly && !heldIds.has(r.ticker_id)) return false;
      if (!q) return true;
      return (
        r.symbol.toLowerCase().includes(q) ||
        (r.name ?? "").toLowerCase().includes(q) ||
        (r.sector ?? "").toLowerCase().includes(q)
      );
    });
  }, [data, filter, heldOnly, heldIds]);

  if (authLoading || !session) {
    return (
      <main className="flex flex-1 items-center justify-center text-sm text-muted">
        Loading…
      </main>
    );
  }

  // Rank position is based on the full ranked list, not the filtered view.
  const rankOf = new Map((data?.rows ?? []).map((r, i) => [r.ticker_id, i + 1]));

  return (
    <>
      <AppHeader />
      <main className="mx-auto w-full max-w-5xl flex-1 px-6 py-8">
        <div className="mb-6 flex flex-wrap items-end justify-between gap-4">
          <div>
            <h1 className="text-xl font-semibold tracking-tight">Screener</h1>
            <p className="mt-1 text-sm text-muted">
              Every covered ticker ranked by relative strength vs the universe.
              {data?.as_of_date ? ` As of ${data.as_of_date}.` : ""}
            </p>
          </div>
          <div className="flex rounded-lg border border-border p-0.5 text-sm">
            {HORIZONS.map((h) => (
              <button
                key={h}
                onClick={() => setHorizon(h)}
                className={`rounded-md px-3 py-1.5 transition-colors ${
                  horizon === h
                    ? "bg-surface-2 text-foreground"
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
            className="flex-1 rounded-lg border border-border bg-surface px-3 py-2 text-sm outline-none focus:border-accent"
          />
          <label className="flex items-center gap-2 text-xs text-muted">
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
          <p className="rounded-lg border border-down/40 bg-down/10 px-3 py-2 text-sm text-down">
            {error}
          </p>
        )}

        {loading ? (
          <div className="rounded-xl border border-border p-10 text-center text-sm text-muted">
            Loading rankings…
          </div>
        ) : visible.length === 0 ? (
          <div className="rounded-xl border border-dashed border-border p-10 text-center text-sm text-muted">
            No tickers match.
          </div>
        ) : (
          <div className="overflow-hidden rounded-xl border border-border">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-border text-left text-xs uppercase tracking-wider text-faint">
                  <th className="w-12 px-4 py-3 text-right font-medium">#</th>
                  <th className="px-4 py-3 font-medium">Ticker</th>
                  <th className="px-4 py-3 font-medium">Relative strength</th>
                  <th className="px-4 py-3 text-right font-medium">Percentile</th>
                </tr>
              </thead>
              <tbody>
                {visible.map((r) => {
                  const pctile = Math.round(r.percentile_rank * 100);
                  const held = heldIds.has(r.ticker_id);
                  return (
                    <tr
                      key={r.ticker_id}
                      onClick={() => router.push(`/ticker/${r.symbol}`)}
                      className={`cursor-pointer border-b border-border/60 last:border-0 transition-colors hover:bg-surface ${
                        held ? "bg-accent/5" : ""
                      }`}
                    >
                      <td className="nums px-4 py-3 text-right text-faint">
                        {rankOf.get(r.ticker_id)}
                      </td>
                      <td className="px-4 py-3">
                        <div className="flex items-center gap-2">
                          <span className="font-medium">{r.symbol}</span>
                          {held && (
                            <span className="rounded bg-accent/15 px-1.5 py-0.5 text-[10px] uppercase tracking-wide text-accent">
                              held
                            </span>
                          )}
                        </div>
                        <div className="max-w-[16rem] truncate text-xs text-muted">
                          {r.name ?? r.sector ?? ""}
                        </div>
                      </td>
                      <td className="px-4 py-3">
                        <div className="h-2 w-full max-w-[14rem] overflow-hidden rounded-full bg-surface-2">
                          <div
                            className={`h-full rounded-full ${barTone(r.percentile_rank)}`}
                            style={{ width: `${pctile}%` }}
                          />
                        </div>
                      </td>
                      <td className="nums px-4 py-3 text-right">{pctile}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}

        <div className="mt-6">
          <Link href="/" className="text-xs text-faint hover:text-foreground">
            ← Back to portfolio
          </Link>
        </div>
      </main>
    </>
  );
}

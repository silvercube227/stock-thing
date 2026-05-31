"use client";

import { useEffect, useMemo, useState } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { useAuth } from "@/components/AuthProvider";
import { AppHeader } from "@/components/AppHeader";
import { getRankings, type RankingResponse } from "@/lib/api";
import { percentileRank } from "@/lib/format";
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

  const rankOf = new Map((data?.rows ?? []).map((r, i) => [r.ticker_id, i + 1]));

  return (
    <>
      <AppHeader />
      <main className="mx-auto w-full max-w-5xl flex-1 px-6 py-8">
        <div className="mb-6 flex flex-wrap items-end justify-between gap-4">
          <div>
            <h1 className="text-xl font-semibold tracking-tight">Screener</h1>
            <p className="mt-1 text-sm text-muted">
              Universe ranked by relative strength.
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
                  <th className="w-12 px-5 py-3.5 text-right font-medium">#</th>
                  <th className="px-5 py-3.5 font-medium">Ticker</th>
                  <th className="px-5 py-3.5 font-medium">Relative strength</th>
                  <th className="px-5 py-3.5 text-right font-medium">Percentile</th>
                </tr>
              </thead>
              <tbody>
                {visible.map((r) => {
                  const pctile = percentileRank(r.percentile_rank);
                  const held = heldIds.has(r.ticker_id);
                  return (
                    <tr
                      key={r.ticker_id}
                      onClick={() => router.push(`/ticker/${r.symbol}`)}
                      className={`cursor-pointer border-b border-border/40 last:border-0 transition-colors hover:bg-surface ${
                        held ? "bg-accent/4" : ""
                      }`}
                    >
                      <td className="nums px-5 py-4 text-right text-faint text-xs">
                        {rankOf.get(r.ticker_id)}
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
                          {r.name ?? r.sector ?? ""}
                        </div>
                      </td>
                      <td className="px-5 py-4">
                        <div className="h-1.5 w-full max-w-[14rem] overflow-hidden rounded-full bg-surface-2">
                          <div
                            className={`h-full rounded-full ${barTone(r.percentile_rank)}`}
                            style={{ width: `${r.percentile_rank * 100}%` }}
                          />
                        </div>
                      </td>
                      <td className="nums px-5 py-4 text-right font-medium">{pctile}</td>
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

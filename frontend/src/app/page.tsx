"use client";

import { useEffect, useMemo } from "react";
import { useRouter } from "next/navigation";
import { useAuth } from "@/components/AuthProvider";
import { AppHeader } from "@/components/AppHeader";
import { NetValueHeader } from "@/components/NetValueHeader";
import { PortfolioTable } from "@/components/PortfolioTable";
import { AddTickerControl } from "@/components/AddTickerControl";
import { usePortfolio } from "@/hooks/usePortfolio";
import { useQuotes } from "@/hooks/useQuotes";

export default function DashboardPage() {
  const router = useRouter();
  const { session, loading: authLoading } = useAuth();
  const { rows, loading, error, add, setShares, remove } = usePortfolio();

  useEffect(() => {
    if (!authLoading && !session) router.replace("/login");
  }, [authLoading, session, router]);

  const symbols = useMemo(() => rows.map((r) => r.symbol), [rows]);
  const quotes = useQuotes(symbols);
  const existingIds = useMemo(
    () => new Set(rows.map((r) => r.ticker_id)),
    [rows],
  );

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
        <NetValueHeader rows={rows} quotes={quotes} />

        <div className="mt-8">
          <h2 className="mb-3 text-sm font-medium text-muted">Add to portfolio</h2>
          <AddTickerControl onAdd={add} existingTickerIds={existingIds} />
        </div>

        <div className="mt-8">
          <h2 className="mb-3 text-sm font-medium text-muted">Holdings</h2>
          {error && (
            <p className="mb-3 rounded-lg border border-down/40 bg-down/10 px-3 py-2 text-sm text-down">
              {error}
            </p>
          )}
          {loading ? (
            <div className="rounded-xl border border-border p-10 text-center text-sm text-muted">
              Loading holdings…
            </div>
          ) : (
            <PortfolioTable
              rows={rows}
              quotes={quotes}
              onSetShares={setShares}
              onRemove={remove}
            />
          )}
        </div>
      </main>
    </>
  );
}

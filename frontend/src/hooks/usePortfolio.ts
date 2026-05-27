"use client";

import { useCallback, useEffect, useState } from "react";
import {
  deleteHolding,
  getPortfolio,
  patchShares,
  upsertHolding,
  type PortfolioRow,
} from "@/lib/api";

export function usePortfolio() {
  const [rows, setRows] = useState<PortfolioRow[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    try {
      setError(null);
      setRows(await getPortfolio());
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to load portfolio");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  const add = useCallback(
    async (symbol: string, shares: number, costBasis?: number | null) => {
      await upsertHolding(symbol, shares, costBasis);
      await refresh();
    },
    [refresh],
  );

  const setShares = useCallback(
    async (tickerId: number, shares: number) => {
      // Optimistic: reflect immediately, then reconcile.
      setRows((prev) =>
        prev.map((r) => (r.ticker_id === tickerId ? { ...r, shares } : r)),
      );
      await patchShares(tickerId, shares);
      await refresh();
    },
    [refresh],
  );

  const remove = useCallback(
    async (tickerId: number) => {
      setRows((prev) => prev.filter((r) => r.ticker_id !== tickerId));
      await deleteHolding(tickerId);
      await refresh();
    },
    [refresh],
  );

  return { rows, loading, error, refresh, add, setShares, remove };
}

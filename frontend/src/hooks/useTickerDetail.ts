"use client";

import { useEffect, useState } from "react";
import {
  getPrices,
  getTickerDetail,
  type PricePoint,
  type TickerDetail,
} from "@/lib/api";

export function useTickerDetail(symbol: string, lookback = "1y") {
  const [detail, setDetail] = useState<TickerDetail | null>(null);
  const [prices, setPrices] = useState<PricePoint[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);
    Promise.all([getTickerDetail(symbol), getPrices(symbol, lookback)])
      .then(([d, p]) => {
        if (cancelled) return;
        setDetail(d);
        setPrices(p);
      })
      .catch((e) => {
        if (!cancelled)
          setError(e instanceof Error ? e.message : "Failed to load ticker");
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [symbol, lookback]);

  return { detail, prices, loading, error };
}

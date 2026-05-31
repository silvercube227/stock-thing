"use client";

import { useCallback, useEffect, useState } from "react";
import {
  getPrices,
  getTickerDetail,
  getTickerStatus,
  type PricePoint,
  type TickerDetail,
} from "@/lib/api";

// Prediction availability for this ticker: "ready" once it has scored rows,
// "running" while a user-added ticker is still being ingested/scored, or a
// terminal "insufficient_history" / "failed" outcome.
export type PredStatus =
  | "ready"
  | "running"
  | "insufficient_history"
  | "failed"
  | "unknown"
  | null;

export function useTickerDetail(symbol: string, lookback = "1y") {
  const [detail, setDetail] = useState<TickerDetail | null>(null);
  const [prices, setPrices] = useState<PricePoint[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [predStatus, setPredStatus] = useState<PredStatus>(null);

  const loadDetail = useCallback(async () => {
    const [d, p] = await Promise.all([
      getTickerDetail(symbol),
      getPrices(symbol, lookback),
    ]);
    setDetail(d);
    setPrices(p);
    return d;
  }, [symbol, lookback]);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);
    setPredStatus(null);
    loadDetail()
      .then(async (d) => {
        if (cancelled) return;
        if (d.predictions.length > 0) {
          setPredStatus("ready");
          return;
        }
        // No predictions yet — ask whether a scoring job is in flight.
        const s = await getTickerStatus(symbol);
        if (!cancelled) setPredStatus(s.status as PredStatus);
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
  }, [symbol, loadDetail]);

  // While a scoring job is running, poll until it resolves, then refetch detail.
  useEffect(() => {
    if (predStatus !== "running") return;
    let cancelled = false;
    const id = setInterval(async () => {
      try {
        const s = await getTickerStatus(symbol);
        if (cancelled) return;
        if (s.status === "ready") {
          await loadDetail();
          if (!cancelled) setPredStatus("ready");
        } else if (s.status !== "running") {
          setPredStatus(s.status as PredStatus); // insufficient_history | failed
        }
      } catch {
        /* transient — keep polling */
      }
    }, 2500);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, [predStatus, symbol, loadDetail]);

  return { detail, prices, loading, error, predStatus };
}

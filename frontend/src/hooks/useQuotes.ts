"use client";

import { useEffect, useRef, useState } from "react";
import { getQuotes, type Quote } from "@/lib/api";

const POLL_MS = 45_000;

/**
 * Poll intraday quotes for the given symbols every ~45s, and on tab focus.
 * Returns a symbol -> Quote map (keys upper-cased by the backend).
 */
export function useQuotes(symbols: string[]) {
  const [quotes, setQuotes] = useState<Record<string, Quote>>({});
  // Stable key so the effect only re-subscribes when the symbol set changes.
  const key = [...symbols].map((s) => s.toUpperCase()).sort().join(",");
  const symbolsRef = useRef<string[]>(symbols);
  symbolsRef.current = symbols;

  useEffect(() => {
    if (!key) {
      setQuotes({});
      return;
    }
    let cancelled = false;

    const tick = async () => {
      try {
        const data = await getQuotes(symbolsRef.current);
        if (!cancelled) setQuotes(data);
      } catch {
        /* transient; keep the last good values */
      }
    };

    tick();
    const id = setInterval(tick, POLL_MS);
    const onFocus = () => tick();
    window.addEventListener("focus", onFocus);

    return () => {
      cancelled = true;
      clearInterval(id);
      window.removeEventListener("focus", onFocus);
    };
  }, [key]);

  return quotes;
}

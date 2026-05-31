"use client";

import { useEffect, useState } from "react";
import { getPrices, type PricePoint } from "@/lib/api";

export function useSparklines(symbols: string[]): Record<string, PricePoint[]> {
  const [sparklines, setSparklines] = useState<Record<string, PricePoint[]>>({});
  const key = symbols.slice().sort().join(",");

  useEffect(() => {
    if (symbols.length === 0) return;
    let cancelled = false;
    Promise.all(
      symbols.map((s) =>
        getPrices(s, "1m")
          .then((data): [string, PricePoint[]] => [s, data])
          .catch((): [string, PricePoint[]] => [s, []]),
      ),
    ).then((entries) => {
      if (!cancelled) setSparklines(Object.fromEntries(entries));
    });
    return () => { cancelled = true; };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key]);

  return sparklines;
}

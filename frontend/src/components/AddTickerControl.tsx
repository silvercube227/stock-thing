"use client";

import { useEffect, useRef, useState } from "react";
import { searchTickers, type TickerSummary } from "@/lib/api";

/**
 * Search the seeded universe (the model only covers tickers already in the DB)
 * and add the selected one to the portfolio.
 */
export function AddTickerControl({
  onAdd,
  existingTickerIds,
}: {
  onAdd: (symbol: string, shares: number, costBasis?: number | null) => Promise<void>;
  existingTickerIds: Set<number>;
}) {
  const [query, setQuery] = useState("");
  const [results, setResults] = useState<TickerSummary[]>([]);
  const [selected, setSelected] = useState<TickerSummary | null>(null);
  const [shares, setShares] = useState("");
  const [cost, setCost] = useState("");
  const [open, setOpen] = useState(false);
  const [busy, setBusy] = useState(false);
  const boxRef = useRef<HTMLDivElement>(null);

  // Debounced catalog search.
  useEffect(() => {
    if (selected || query.trim().length === 0) {
      setResults([]);
      return;
    }
    const id = setTimeout(async () => {
      try {
        setResults(await searchTickers(query.trim()));
        setOpen(true);
      } catch {
        setResults([]);
      }
    }, 200);
    return () => clearTimeout(id);
  }, [query, selected]);

  // Close the dropdown on outside click.
  useEffect(() => {
    function onClick(e: MouseEvent) {
      if (boxRef.current && !boxRef.current.contains(e.target as Node)) {
        setOpen(false);
      }
    }
    document.addEventListener("mousedown", onClick);
    return () => document.removeEventListener("mousedown", onClick);
  }, []);

  function pick(t: TickerSummary) {
    setSelected(t);
    setQuery(t.symbol);
    setOpen(false);
  }

  function reset() {
    setSelected(null);
    setQuery("");
    setShares("");
    setCost("");
    setResults([]);
  }

  async function submit() {
    if (!selected) return;
    const n = Number(shares);
    if (Number.isNaN(n) || n <= 0) return;
    const c = cost.trim() === "" ? null : Number(cost);
    if (c !== null && (Number.isNaN(c) || c < 0)) return;
    setBusy(true);
    try {
      await onAdd(selected.symbol, n, c);
      reset();
    } finally {
      setBusy(false);
    }
  }

  return (
    <div ref={boxRef} className="relative">
      <div className="flex gap-2">
        <div className="relative flex-1">
          <input
            value={query}
            placeholder="Search ticker to add (e.g. AAPL)…"
            onChange={(e) => {
              setQuery(e.target.value);
              setSelected(null);
            }}
            onFocus={() => results.length && setOpen(true)}
            className="w-full rounded-lg border border-border bg-surface px-3 py-2 text-sm outline-none focus:border-accent"
          />
          {open && results.length > 0 && (
            <ul className="absolute z-10 mt-1 max-h-72 w-full overflow-auto rounded-lg border border-border bg-surface-2 py-1 shadow-xl">
              {results.map((t) => {
                const held = existingTickerIds.has(t.ticker_id);
                return (
                  <li key={t.ticker_id}>
                    <button
                      disabled={held}
                      onClick={() => pick(t)}
                      className="flex w-full items-center justify-between px-3 py-2 text-left text-sm hover:bg-surface disabled:opacity-40"
                    >
                      <span>
                        <span className="font-medium">{t.symbol}</span>
                        <span className="ml-2 text-xs text-muted">{t.name}</span>
                      </span>
                      <span className="text-xs text-faint">
                        {held ? "held" : t.sector}
                      </span>
                    </button>
                  </li>
                );
              })}
            </ul>
          )}
        </div>

        <input
          type="number"
          min={0}
          step="any"
          value={shares}
          placeholder="Shares"
          onChange={(e) => setShares(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && submit()}
          className="nums w-24 rounded-lg border border-border bg-surface px-3 py-2 text-right text-sm outline-none focus:border-accent"
        />
        <input
          type="number"
          min={0}
          step="any"
          value={cost}
          placeholder="Cost/sh"
          title="Optional cost per share, for total-return tracking"
          onChange={(e) => setCost(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && submit()}
          className="nums w-24 rounded-lg border border-border bg-surface px-3 py-2 text-right text-sm outline-none focus:border-accent"
        />
        <button
          onClick={submit}
          disabled={!selected || !shares || busy}
          className="rounded-lg bg-accent px-4 py-2 text-sm font-medium text-background transition-opacity hover:opacity-90 disabled:opacity-40"
        >
          Add
        </button>
      </div>
      <p className="mt-2 text-xs text-faint">
        Limited to covered tickers (the model&apos;s tracked universe).
      </p>
    </div>
  );
}

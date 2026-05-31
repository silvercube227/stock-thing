"use client";

import { useEffect, useRef, useState } from "react";
import { searchTickers, type TickerSummary } from "@/lib/api";

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
            placeholder="Search ticker… (e.g. AAPL)"
            onChange={(e) => {
              setQuery(e.target.value);
              setSelected(null);
            }}
            onFocus={() => results.length && setOpen(true)}
            className="w-full rounded-xl border border-border bg-surface px-4 py-2.5 text-sm outline-none transition-colors placeholder:text-faint focus:border-accent/60"
          />
          {open && results.length > 0 && (
            <ul className="absolute z-10 mt-1 max-h-72 w-full overflow-auto rounded-xl border border-border bg-surface py-1 shadow-2xl shadow-black/60">
              {results.map((t) => {
                const held = existingTickerIds.has(t.ticker_id);
                return (
                  <li key={t.ticker_id}>
                    <button
                      disabled={held}
                      onClick={() => pick(t)}
                      className="flex w-full items-center justify-between px-4 py-2.5 text-left text-sm transition-colors hover:bg-surface-2 disabled:opacity-40"
                    >
                      <span>
                        <span className="font-semibold tracking-wide">{t.symbol}</span>
                        <span className="ml-2 text-xs text-muted">{t.name}</span>
                      </span>
                      <span className="text-[11px] text-faint">
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
          className="nums w-24 rounded-xl border border-border bg-surface px-3 py-2.5 text-right text-sm outline-none transition-colors placeholder:text-faint focus:border-accent/60"
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
          className="nums w-24 rounded-xl border border-border bg-surface px-3 py-2.5 text-right text-sm outline-none transition-colors placeholder:text-faint focus:border-accent/60"
        />
        <button
          onClick={submit}
          disabled={!selected || !shares || busy}
          className="rounded-xl bg-accent px-5 py-2.5 text-sm font-semibold text-background transition-opacity hover:opacity-90 disabled:opacity-30"
        >
          {busy ? "…" : "Add"}
        </button>
      </div>
      <p className="mt-2 text-[11px] text-faint">
        Limited to covered tickers (the model&apos;s tracked universe).
      </p>
    </div>
  );
}

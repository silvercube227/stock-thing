"use client";

import { useRouter } from "next/navigation";
import type { PortfolioRow, Quote, PricePoint } from "@/lib/api";
import { changeColor, money, pct } from "@/lib/format";
import { SharesEditor } from "./SharesEditor";
import { Sparkline } from "./Sparkline";

export function PortfolioTable({
  rows,
  quotes,
  sparklines,
  onSetShares,
  onRemove,
}: {
  rows: PortfolioRow[];
  quotes: Record<string, Quote>;
  sparklines: Record<string, PricePoint[]>;
  onSetShares: (tickerId: number, shares: number) => void;
  onRemove: (tickerId: number) => void;
}) {
  const router = useRouter();

  if (rows.length === 0) {
    return (
      <div className="rounded-2xl border border-dashed border-border p-12 text-center text-sm text-muted">
        No holdings yet. Add a ticker to start tracking.
      </div>
    );
  }

  return (
    <div className="overflow-hidden rounded-2xl border border-border">
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b border-border bg-surface-2 text-left text-xs text-muted">
            <th className="px-5 py-3.5 font-medium">Ticker</th>
            <th className="px-5 py-3.5 font-medium">1M</th>
            <th className="px-5 py-3.5 text-right font-medium">Price</th>
            <th className="px-5 py-3.5 text-right font-medium">Day</th>
            <th className="px-5 py-3.5 text-right font-medium">Shares</th>
            <th className="px-5 py-3.5 text-right font-medium">Value</th>
            <th className="px-5 py-3.5" />
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => {
            const q = quotes[row.symbol.toUpperCase()];
            const price = q?.price ?? row.last_close;
            const value = price !== null ? price * row.shares : null;
            return (
              <tr
                key={row.ticker_id}
                onClick={() => router.push(`/ticker/${row.symbol}`)}
                className="cursor-pointer border-b border-border/40 last:border-0 transition-colors hover:bg-surface"
              >
                <td className="px-5 py-4">
                  <div className="font-semibold tracking-wide">{row.symbol}</div>
                  <div className="max-w-[14rem] truncate text-[11px] text-muted mt-0.5">
                    {row.name ?? row.sector ?? ""}
                  </div>
                </td>
                <td className="px-5 py-4">
                  <Sparkline data={sparklines[row.symbol] ?? []} />
                </td>
                <td className="nums px-5 py-4 text-right font-medium">
                  {money(price)}
                  {q?.stale && (
                    <span
                      title="Last stored close (live quote unavailable)"
                      className="ml-1 text-faint"
                    >
                      ·
                    </span>
                  )}
                </td>
                <td className={`nums px-5 py-4 text-right font-medium ${changeColor(q?.change_pct)}`}>
                  {q?.change_pct != null ? pct(q.change_pct) : "—"}
                </td>
                <td className="px-5 py-4 text-right">
                  <SharesEditor
                    value={row.shares}
                    onCommit={(s) => onSetShares(row.ticker_id, s)}
                  />
                </td>
                <td className="nums px-5 py-4 text-right font-medium">{money(value)}</td>
                <td className="px-5 py-4 text-right">
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      onRemove(row.ticker_id);
                    }}
                    className="text-xs text-faint transition-colors hover:text-down"
                    title="Remove holding"
                  >
                    ✕
                  </button>
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

"use client";

import { useRouter } from "next/navigation";
import type { PortfolioRow, Quote } from "@/lib/api";
import { changeColor, money, pct } from "@/lib/format";
import { SharesEditor } from "./SharesEditor";

export function PortfolioTable({
  rows,
  quotes,
  onSetShares,
  onRemove,
}: {
  rows: PortfolioRow[];
  quotes: Record<string, Quote>;
  onSetShares: (tickerId: number, shares: number) => void;
  onRemove: (tickerId: number) => void;
}) {
  const router = useRouter();

  if (rows.length === 0) {
    return (
      <div className="rounded-xl border border-dashed border-border p-10 text-center text-sm text-muted">
        No holdings yet. Add a ticker to start tracking.
      </div>
    );
  }

  return (
    <div className="overflow-hidden rounded-xl border border-border">
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b border-border text-left text-xs uppercase tracking-wider text-faint">
            <th className="px-4 py-3 font-medium">Ticker</th>
            <th className="px-4 py-3 text-right font-medium">Price</th>
            <th className="px-4 py-3 text-right font-medium">Day</th>
            <th className="px-4 py-3 text-right font-medium">Shares</th>
            <th className="px-4 py-3 text-right font-medium">Value</th>
            <th className="px-4 py-3" />
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
                className="cursor-pointer border-b border-border/60 last:border-0 transition-colors hover:bg-surface"
              >
                <td className="px-4 py-3">
                  <div className="font-medium">{row.symbol}</div>
                  <div className="max-w-[14rem] truncate text-xs text-muted">
                    {row.name ?? row.sector ?? ""}
                  </div>
                </td>
                <td className="nums px-4 py-3 text-right">
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
                <td className={`nums px-4 py-3 text-right ${changeColor(q?.change_pct)}`}>
                  {q?.change_pct != null ? pct(q.change_pct) : "—"}
                </td>
                <td className="px-4 py-3 text-right">
                  <SharesEditor
                    value={row.shares}
                    onCommit={(s) => onSetShares(row.ticker_id, s)}
                  />
                </td>
                <td className="nums px-4 py-3 text-right">{money(value)}</td>
                <td className="px-4 py-3 text-right">
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      onRemove(row.ticker_id);
                    }}
                    className="text-xs text-faint hover:text-down"
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

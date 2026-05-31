"use client";

import type { PortfolioRow, Quote } from "@/lib/api";
import { changeColor, money, pct } from "@/lib/format";

function priceFor(row: PortfolioRow, quotes: Record<string, Quote>): number | null {
  const q = quotes[row.symbol.toUpperCase()];
  return q?.price ?? row.last_close;
}

export function NetValueHeader({
  rows,
  quotes,
}: {
  rows: PortfolioRow[];
  quotes: Record<string, Quote>;
}) {
  let total = 0;
  let dayChange = 0;
  let anyPrice = false;

  let costBasisValue = 0;
  let totalCost = 0;

  for (const row of rows) {
    const price = priceFor(row, quotes);
    if (price !== null) {
      total += price * row.shares;
      anyPrice = true;
      if (row.cost_basis != null) {
        costBasisValue += price * row.shares;
        totalCost += row.cost_basis * row.shares;
      }
    }
    const q = quotes[row.symbol.toUpperCase()];
    if (q?.change != null) dayChange += q.change * row.shares;
  }

  const prior = total - dayChange;
  const dayPct = prior > 0 ? (dayChange / prior) * 100 : null;

  const totalReturn = totalCost > 0 ? costBasisValue - totalCost : null;
  const totalReturnPct = totalCost > 0 ? (totalReturn! / totalCost) * 100 : null;
  const partialCostCoverage =
    totalCost > 0 && rows.some((r) => r.cost_basis == null);

  return (
    <div className="rounded-2xl border border-border bg-surface p-6">
      <div className="text-xs text-muted">Portfolio value</div>

      <div className="mt-2 flex flex-wrap items-baseline gap-x-4 gap-y-1">
        <span className="nums text-4xl font-semibold">
          {anyPrice ? money(total) : "—"}
        </span>
        {dayChange !== 0 && (
          <span className={`nums text-sm font-medium ${changeColor(dayChange)}`}>
            {dayChange >= 0 ? "+" : ""}{money(dayChange)}{" "}
            <span className="text-xs opacity-80">({pct(dayPct)}) today</span>
          </span>
        )}
      </div>

      <div className="mt-4 flex flex-wrap items-center gap-x-6 gap-y-2 border-t border-border/60 pt-4">
        {totalReturn !== null && (
          <div className="flex items-baseline gap-2 text-sm">
            <span className="text-faint text-xs uppercase tracking-wider">Total return</span>
            <span className={`nums font-medium ${changeColor(totalReturn)}`}>
              {money(totalReturn)} ({pct(totalReturnPct)})
            </span>
            {partialCostCoverage && (
              <span title="Only holdings with a recorded cost basis are included." className="text-faint">
                ·
              </span>
            )}
          </div>
        )}
        <div className="text-xs text-faint">
          {rows.length} {rows.length === 1 ? "holding" : "holdings"}
        </div>
      </div>
    </div>
  );
}

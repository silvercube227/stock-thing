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

  // Total return is computed only over holdings that have BOTH a cost basis and a
  // current price, so the figure is honest about what it covers.
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
    <div className="rounded-xl border border-border bg-surface p-6">
      <div className="text-xs uppercase tracking-wider text-faint">
        Portfolio value
      </div>
      <div className="mt-1 flex flex-wrap items-baseline gap-x-4 gap-y-1">
        <span className="nums text-4xl font-semibold">
          {anyPrice ? money(total) : "—"}
        </span>
        {dayChange !== 0 && (
          <span className={`nums text-sm ${changeColor(dayChange)}`}>
            {money(dayChange)} ({pct(dayPct)}) today
          </span>
        )}
      </div>
      {totalReturn !== null && (
        <div className="mt-2 flex items-baseline gap-2 text-sm">
          <span className="text-faint">Total return</span>
          <span className={`nums ${changeColor(totalReturn)}`}>
            {money(totalReturn)} ({pct(totalReturnPct)})
          </span>
          {partialCostCoverage && (
            <span
              title="Only holdings with a recorded cost basis are included."
              className="text-faint"
            >
              ·
            </span>
          )}
        </div>
      )}
      <div className="mt-1 text-xs text-faint">
        {rows.length} {rows.length === 1 ? "holding" : "holdings"}
      </div>
    </div>
  );
}

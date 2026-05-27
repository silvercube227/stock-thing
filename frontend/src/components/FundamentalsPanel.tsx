"use client";

import type { FundamentalsSnapshot } from "@/lib/api";
import { compactMoney, num } from "@/lib/format";

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-lg border border-border bg-surface px-3 py-2.5">
      <div className="text-[11px] uppercase tracking-wider text-faint">{label}</div>
      <div className="nums mt-0.5 text-sm">{value}</div>
    </div>
  );
}

function ratioPct(v: number | null | undefined): string {
  if (v === null || v === undefined) return "—";
  return `${(v * 100).toFixed(1)}%`;
}

export function FundamentalsPanel({ f }: { f: FundamentalsSnapshot | null }) {
  if (!f) {
    return <p className="text-sm text-muted">No fundamentals on file.</p>;
  }

  const debtEquity =
    f.total_debt != null && f.total_equity
      ? num(f.total_debt / f.total_equity)
      : "—";

  return (
    <div>
      <div className="grid grid-cols-2 gap-2 sm:grid-cols-3">
        <Metric label="Revenue" value={compactMoney(f.revenue)} />
        <Metric label="Net income" value={compactMoney(f.net_income)} />
        <Metric label="Free cash flow" value={compactMoney(f.fcf)} />
        <Metric label="Gross margin" value={ratioPct(f.gross_margin)} />
        <Metric label="Operating margin" value={ratioPct(f.operating_margin)} />
        <Metric label="Debt / equity" value={debtEquity} />
      </div>
      <p className="mt-3 text-[11px] text-faint">
        {f.filing_type ?? "Latest filing"}
        {f.period_end ? ` · period ending ${f.period_end}` : ""}
        {f.filed_at ? ` · filed ${f.filed_at}` : ""}
      </p>
    </div>
  );
}

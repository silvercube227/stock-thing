"use client";

import type { FundamentalsSnapshot, ValuationSnapshot } from "@/lib/api";
import { compactMoney, num } from "@/lib/format";

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <div className="text-[10px] uppercase tracking-widest text-faint">{label}</div>
      <div className="nums mt-0.5 text-sm font-medium">{value}</div>
    </div>
  );
}

function ratioPct(v: number | null | undefined): string {
  if (v === null || v === undefined) return "—";
  return `${(v * 100).toFixed(1)}%`;
}

export function FundamentalsPanel({
  f,
  valuation,
}: {
  f: FundamentalsSnapshot | null;
  valuation?: ValuationSnapshot | null;
}) {
  const hasVal =
    !!valuation &&
    (valuation.trailing_pe != null ||
      valuation.forward_pe != null ||
      valuation.price_to_sales != null ||
      valuation.ebitda != null);

  // EDGAR is primary. For off-index names with no SEC filing, fall back to the
  // yfinance .info fundamentals (debt/equity is left out — yfinance's scale for
  // it is unreliable).
  const yfHasFund =
    !!valuation &&
    (valuation.revenue != null ||
      valuation.net_income != null ||
      valuation.fcf != null ||
      valuation.gross_margin != null ||
      valuation.operating_margin != null);

  const fnd = f
    ? {
        revenue: f.revenue,
        net_income: f.net_income,
        fcf: f.fcf,
        gross_margin: f.gross_margin,
        operating_margin: f.operating_margin,
        debtEquity:
          f.total_debt != null && f.total_equity
            ? num(f.total_debt / f.total_equity)
            : "—",
      }
    : yfHasFund
      ? {
          revenue: valuation!.revenue,
          net_income: valuation!.net_income,
          fcf: valuation!.fcf,
          gross_margin: valuation!.gross_margin,
          operating_margin: valuation!.operating_margin,
          debtEquity: "—",
        }
      : null;

  if (!fnd && !hasVal) {
    return <p className="text-sm text-muted">No fundamentals on file.</p>;
  }

  const yfSourced = !f; // any rendered figures came from yfinance, not EDGAR

  return (
    <div>
      {fnd && (
        <div className="grid grid-cols-2 gap-x-6 gap-y-4 sm:grid-cols-3">
          <Metric label="Revenue" value={compactMoney(fnd.revenue)} />
          <Metric label="Net income" value={compactMoney(fnd.net_income)} />
          <Metric label="Free cash flow" value={compactMoney(fnd.fcf)} />
          <Metric label="Gross margin" value={ratioPct(fnd.gross_margin)} />
          <Metric label="Operating margin" value={ratioPct(fnd.operating_margin)} />
          <Metric label="Debt / equity" value={fnd.debtEquity} />
        </div>
      )}

      {hasVal && (
        <div className={fnd ? "mt-4" : ""}>
          <div className="grid grid-cols-2 gap-x-6 gap-y-4 sm:grid-cols-3">
            <Metric label="P / E (TTM)" value={num(valuation!.trailing_pe, 1)} />
            <Metric label="P / E (fwd)" value={num(valuation!.forward_pe, 1)} />
            <Metric label="P / S (TTM)" value={num(valuation!.price_to_sales, 1)} />
            <Metric label="EBITDA" value={compactMoney(valuation!.ebitda)} />
          </div>
        </div>
      )}

      <p className="mt-4 text-[10px] text-faint">
        {f ? (
          <>
            {f.filing_type ?? "Latest filing"}
            {f.period_end ? ` · period ending ${f.period_end}` : ""}
            {f.filed_at ? ` · filed ${f.filed_at}` : ""}
            {hasVal ? " · P/E · P/S · EBITDA live from yfinance" : ""}
          </>
        ) : (
          yfSourced && (fnd || hasVal) && "Live from yfinance (no SEC filing on record)"
        )}
      </p>
    </div>
  );
}

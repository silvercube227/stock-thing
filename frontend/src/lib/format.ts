export function money(n: number | null | undefined, dp = 2): string {
  if (n === null || n === undefined || Number.isNaN(n)) return "—";
  return n.toLocaleString("en-US", {
    style: "currency",
    currency: "USD",
    minimumFractionDigits: dp,
    maximumFractionDigits: dp,
  });
}

export function pct(n: number | null | undefined, dp = 2): string {
  if (n === null || n === undefined || Number.isNaN(n)) return "—";
  const sign = n > 0 ? "+" : "";
  return `${sign}${n.toFixed(dp)}%`;
}

export function num(n: number | null | undefined, dp = 2): string {
  if (n === null || n === undefined || Number.isNaN(n)) return "—";
  return n.toLocaleString("en-US", {
    minimumFractionDigits: dp,
    maximumFractionDigits: dp,
  });
}

// Compact large dollar figures, e.g. revenue: $1.2B, $340M.
export function compactMoney(n: number | null | undefined): string {
  if (n === null || n === undefined || Number.isNaN(n)) return "—";
  return n.toLocaleString("en-US", {
    style: "currency",
    currency: "USD",
    notation: "compact",
    maximumFractionDigits: 1,
  });
}

export function changeColor(n: number | null | undefined): string {
  if (n === null || n === undefined || n === 0) return "text-muted";
  return n > 0 ? "text-up" : "text-down";
}

/** Cross-sectional rank on a 0–100 scale (input is 0–1). */
export function percentileRank(
  rank: number | null | undefined,
  dp = 1,
): string {
  if (rank === null || rank === undefined || Number.isNaN(rank)) return "—";
  return (rank * 100).toFixed(dp);
}

/** Share of the universe ranked below this name (complement of percentile). */
export function topUniversePct(
  rank: number | null | undefined,
  dp = 1,
): string {
  if (rank === null || rank === undefined || Number.isNaN(rank)) return "—";
  return Math.max(0, (1 - rank) * 100).toFixed(dp);
}

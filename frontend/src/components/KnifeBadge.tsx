"use client";

// Transparency tag for "falling knife" names — high realized volatility AND below
// trend / near the 52-week low. Mirrors the vol×downtrend signal the model uses
// internally, surfaced (not hidden in a rank tweak). Renders nothing for "none".
export function KnifeBadge({ tier }: { tier: string | null | undefined }) {
  if (tier !== "elevated" && tier !== "high") return null;
  const high = tier === "high";
  return (
    <span
      title="High realized volatility AND below trend / near 52-week low — a 'falling knife'. Informational only; it does not change the rank."
      className={`shrink-0 rounded-sm px-1.5 py-0.5 text-[9px] uppercase tracking-widest ${
        high ? "bg-down/20 text-down" : "bg-down/10 text-down/80"
      }`}
    >
      {high ? "⚠ knife" : "elevated vol"}
    </span>
  );
}

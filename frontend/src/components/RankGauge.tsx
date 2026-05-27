"use client";

import type { HorizonPrediction } from "@/lib/api";

function rankTone(rank: number): { bar: string; text: string } {
  if (rank >= 0.66) return { bar: "bg-up", text: "text-up" };
  if (rank <= 0.34) return { bar: "bg-down", text: "text-down" };
  return { bar: "bg-accent", text: "text-accent" };
}

function stabilityLabel(std: number): string {
  if (std < 0.05) return "steady";
  if (std < 0.12) return "moderate";
  return "shifting";
}

function RankGauge({
  horizon,
  rank,
  rankStd,
  lowSignal,
}: {
  horizon: string;
  rank: number;
  rankStd: number | null;
  lowSignal?: boolean;
}) {
  const pctile = Math.round(rank * 100);
  const topPct = Math.max(1, Math.round((1 - rank) * 100));
  const tone = rankTone(rank);

  return (
    <div>
      <div className="mb-1.5 flex items-baseline justify-between">
        <div className="flex items-center gap-2">
          <span className="text-sm font-medium">{horizon}</span>
          {lowSignal && (
            <span
              title="Validation shows little to no skill at the 1-month horizon."
              className="rounded bg-surface-2 px-1.5 py-0.5 text-[10px] uppercase tracking-wide text-faint"
            >
              low signal
            </span>
          )}
        </div>
        <span className={`nums text-sm ${tone.text}`}>top {topPct}%</span>
      </div>
      <div className="h-2 w-full overflow-hidden rounded-full bg-surface-2">
        <div
          className={`h-full rounded-full ${tone.bar} ${lowSignal ? "opacity-40" : ""}`}
          style={{ width: `${pctile}%` }}
        />
      </div>
      <div className="mt-1 flex justify-between text-[11px] text-faint">
        <span>{pctile}th percentile vs universe</span>
        {rankStd != null && (
          <span
            className="nums"
            title="Std of the predicted rank over the last 3 scoring dates — lower is steadier."
          >
            rank σ {rankStd.toFixed(2)} · {stabilityLabel(rankStd)}
          </span>
        )}
      </div>
    </div>
  );
}

export function RankGaugeRow({
  predictions,
  asOf,
}: {
  predictions: HorizonPrediction[];
  asOf: string | null;
}) {
  if (predictions.length === 0) {
    return (
      <p className="text-sm text-muted">
        No model projections available for this ticker yet.
      </p>
    );
  }

  return (
    <div>
      <div className="space-y-4">
        {predictions.map((p) => (
          <RankGauge
            key={p.horizon}
            horizon={p.horizon}
            rank={p.percentile_rank}
            rankStd={p.rank_std}
            lowSignal={p.horizon === "1M"}
          />
        ))}
      </div>
      <p className="mt-4 text-[11px] leading-relaxed text-faint">
        Relative strength vs the tracked universe (cross-sectional percentile rank),
        not a probability of going up.{asOf ? ` As of ${asOf}.` : ""}
      </p>
    </div>
  );
}

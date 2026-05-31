"use client";

import type { HorizonPrediction } from "@/lib/api";
import { percentileRank, topUniversePct } from "@/lib/format";

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
  const pctile = percentileRank(rank);
  const topPct = topUniversePct(rank);
  const tone = rankTone(rank);

  return (
    <div>
      <div className="mb-2 flex items-baseline justify-between">
        <div className="flex items-center gap-2">
          <span className="text-xs font-semibold uppercase tracking-widest text-muted">{horizon}</span>
          {lowSignal && (
            <span
              title="Validation shows little to no skill at the 1-month horizon."
              className="rounded-sm bg-surface-2 px-1.5 py-0.5 text-[9px] uppercase tracking-widest text-faint border border-border"
            >
              low signal
            </span>
          )}
        </div>
        <span className={`nums text-sm font-semibold ${tone.text}`}>top {topPct}%</span>
      </div>
      <div className="h-1.5 w-full overflow-hidden rounded-full bg-surface-2">
        <div
          className={`h-full rounded-full transition-all ${tone.bar} ${lowSignal ? "opacity-30" : ""}`}
          style={{ width: `${rank * 100}%` }}
        />
      </div>
      <div className="mt-1.5 flex justify-between text-[10px] text-faint">
        <span>{pctile} percentile</span>
        {rankStd != null && (
          <span
            className="nums"
            title="Std of the predicted rank over the last 3 scoring dates — lower is steadier."
          >
            σ {rankStd.toFixed(2)} · {stabilityLabel(rankStd)}
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
      <div className="space-y-5">
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
      <p className="mt-5 text-[10px] leading-relaxed text-faint border-t border-border/40 pt-3">
        Cross-sectional percentile rank vs tracked universe — relative strength,
        not a directional probability.{asOf ? ` As of ${asOf}.` : ""}
      </p>
    </div>
  );
}

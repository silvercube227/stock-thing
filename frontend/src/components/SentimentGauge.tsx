"use client";

import type { SentimentSnapshot } from "@/lib/api";

function tone(v: number | null | undefined) {
  if (v == null) return { label: "No data", text: "text-muted", dot: "bg-muted" };
  if (v > 0.05) return { label: "Positive", text: "text-up", dot: "bg-up" };
  if (v < -0.05) return { label: "Negative", text: "text-down", dot: "bg-down" };
  return { label: "Neutral", text: "text-accent", dot: "bg-accent" };
}

export function SentimentGauge({ s }: { s: SentimentSnapshot | null }) {
  if (!s || s.rolling_7d == null) {
    return <p className="text-sm text-muted">No recent sentiment.</p>;
  }

  const v = s.rolling_7d;
  const pos = Math.min(100, Math.max(0, ((v + 1) / 2) * 100));
  const t = tone(v);

  return (
    <div>
      <div className="flex items-baseline justify-between">
        <span className={`text-sm font-semibold ${t.text}`}>{t.label}</span>
        <span className="nums text-sm font-medium">{v.toFixed(2)}</span>
      </div>

      <div className="relative mt-3 h-1.5 rounded-full bg-gradient-to-r from-down/40 via-surface-2 to-up/40">
        <div className="absolute left-1/2 top-1/2 h-3 w-px -translate-x-1/2 -translate-y-1/2 bg-border" />
        <div
          className={`absolute top-1/2 h-3 w-3 -translate-x-1/2 -translate-y-1/2 rounded-full ${t.dot} ring-2 ring-background`}
          style={{ left: `${pos}%` }}
        />
      </div>

      <div className="mt-4 flex justify-between text-[10px] text-faint border-t border-border/40 pt-3">
        <span className="uppercase tracking-widest">7-day rolling</span>
        <span className="nums">
          14d {s.rolling_14d != null ? s.rolling_14d.toFixed(2) : "—"}
          {s.headline_count != null ? ` · ${s.headline_count} headlines` : ""}
        </span>
      </div>
    </div>
  );
}

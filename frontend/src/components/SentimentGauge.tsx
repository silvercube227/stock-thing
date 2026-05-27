"use client";

import type { SentimentSnapshot } from "@/lib/api";

function tone(v: number | null | undefined) {
  if (v == null) return { label: "No data", text: "text-muted", dot: "bg-muted" };
  if (v > 0.05) return { label: "Positive", text: "text-up", dot: "bg-up" };
  if (v < -0.05) return { label: "Negative", text: "text-down", dot: "bg-down" };
  return { label: "Neutral", text: "text-muted", dot: "bg-muted" };
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
        <span className={`text-sm font-medium ${t.text}`}>{t.label}</span>
        <span className="nums text-sm">{v.toFixed(2)}</span>
      </div>

      <div className="relative mt-3 h-2 rounded-full bg-gradient-to-r from-down/50 via-surface-2 to-up/50">
        {/* center reference */}
        <div className="absolute left-1/2 top-1/2 h-3 w-px -translate-x-1/2 -translate-y-1/2 bg-border" />
        {/* current value */}
        <div
          className={`absolute top-1/2 h-3 w-3 -translate-x-1/2 -translate-y-1/2 rounded-full ${t.dot} ring-2 ring-background`}
          style={{ left: `${pos}%` }}
        />
      </div>

      <div className="mt-3 flex justify-between text-[11px] text-faint">
        <span>7-day rolling</span>
        <span className="nums">
          14d {s.rolling_14d != null ? s.rolling_14d.toFixed(2) : "—"}
          {s.headline_count != null ? ` · ${s.headline_count} headlines` : ""}
        </span>
      </div>
    </div>
  );
}

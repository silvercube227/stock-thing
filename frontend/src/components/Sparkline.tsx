"use client";

import type { PricePoint } from "@/lib/api";

export function Sparkline({ data }: { data: PricePoint[] }) {
  if (data.length < 2) return <div className="w-20" />;

  const W = 80;
  const H = 32;
  const prices = data.map((p) => p.close);
  const min = Math.min(...prices);
  const max = Math.max(...prices);
  const range = max - min || 1;

  const points = prices
    .map((p, i) => {
      const x = (i / (prices.length - 1)) * W;
      const y = H - ((p - min) / range) * H;
      return `${x},${y}`;
    })
    .join(" ");

  const up = prices[prices.length - 1] >= prices[0];
  const color = up ? "#10b981" : "#f43f5e";

  return (
    <svg width={W} height={H} viewBox={`0 0 ${W} ${H}`} className="overflow-visible">
      <polyline
        points={points}
        fill="none"
        stroke={color}
        strokeWidth="1.5"
        strokeLinejoin="round"
        strokeLinecap="round"
      />
    </svg>
  );
}

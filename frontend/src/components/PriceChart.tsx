"use client";

import { useEffect, useRef } from "react";
import {
  AreaSeries,
  CandlestickSeries,
  ColorType,
  createChart,
  type IChartApi,
  type ISeriesApi,
} from "lightweight-charts";
import type { PricePoint } from "@/lib/api";
import { useTheme } from "@/components/ThemeProvider";

export type ChartMode = "area" | "candles";

// Resolve a theme design token to its concrete color so the chart (which can't use
// CSS utilities) matches light/dark. Falls back to a sane dark value off-DOM.
function cssVar(name: string, fallback: string): string {
  if (typeof window === "undefined") return fallback;
  const v = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  return v || fallback;
}

export function PriceChart({
  data,
  mode = "area",
}: {
  data: PricePoint[];
  mode?: ChartMode;
}) {
  const containerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const seriesRef = useRef<ISeriesApi<"Area"> | ISeriesApi<"Candlestick"> | null>(null);
  const { theme } = useTheme();

  // Recreate the chart + series when the series type (mode) OR the theme changes;
  // the data effect below repopulates it. Colors are read from the CSS tokens so
  // they track light/dark.
  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;

    const accent = cssVar("--color-accent", "#e11d48");
    const grid = cssVar("--color-surface-2", "#1e0f1b");
    const border = cssVar("--color-border", "#331823");
    const text = cssVar("--color-faint", "#6e4c5c");
    const up = cssVar("--color-up", "#10b981");
    const down = cssVar("--color-down", "#fb7185");

    const chart = createChart(el, {
      width: el.clientWidth,
      height: 300,
      layout: {
        background: { type: ColorType.Solid, color: "transparent" },
        textColor: text,
        fontFamily: "var(--font-geist-mono), monospace",
        attributionLogo: false,
      },
      grid: {
        vertLines: { visible: false },
        horzLines: { color: grid },
      },
      rightPriceScale: { borderColor: border },
      timeScale: { borderColor: border, fixLeftEdge: true, fixRightEdge: true },
      crosshair: {
        horzLine: { labelBackgroundColor: accent },
        vertLine: { labelBackgroundColor: accent },
      },
    });

    const series =
      mode === "candles"
        ? chart.addSeries(CandlestickSeries, {
            upColor: up,
            downColor: down,
            borderUpColor: up,
            borderDownColor: down,
            wickUpColor: up,
            wickDownColor: down,
            priceLineVisible: false,
          })
        : chart.addSeries(AreaSeries, {
            lineColor: accent,
            lineWidth: 2,
            topColor: `${accent}33`,
            bottomColor: `${accent}03`,
            priceLineVisible: false,
          });

    chartRef.current = chart;
    seriesRef.current = series;

    const ro = new ResizeObserver(() => {
      chart.applyOptions({ width: el.clientWidth });
    });
    ro.observe(el);

    return () => {
      ro.disconnect();
      chart.remove();
      chartRef.current = null;
      seriesRef.current = null;
    };
  }, [mode, theme]);

  useEffect(() => {
    const series = seriesRef.current;
    if (!series) return;
    if (mode === "candles") {
      // Candlesticks need full OHLC; older rows may lack it — drop those.
      (series as ISeriesApi<"Candlestick">).setData(
        data
          .filter((p) => p.open != null && p.high != null && p.low != null)
          .map((p) => ({
            time: p.date,
            open: p.open as number,
            high: p.high as number,
            low: p.low as number,
            close: p.close,
          })),
      );
    } else {
      (series as ISeriesApi<"Area">).setData(
        data.map((p) => ({ time: p.date, value: p.close })),
      );
    }
    chartRef.current?.timeScale().fitContent();
  }, [data, mode, theme]);

  return <div ref={containerRef} className="w-full" />;
}

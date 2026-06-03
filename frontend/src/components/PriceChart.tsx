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

export type ChartMode = "area" | "candles";

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

  // Recreate the chart + series when the series type (mode) changes; the data
  // effect below repopulates it.
  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;

    const chart = createChart(el, {
      width: el.clientWidth,
      height: 300,
      layout: {
        background: { type: ColorType.Solid, color: "transparent" },
        textColor: "#384d6e",
        fontFamily: "var(--font-geist-mono), monospace",
        attributionLogo: false,
      },
      grid: {
        vertLines: { visible: false },
        horzLines: { color: "#0d1829" },
      },
      rightPriceScale: { borderColor: "#162038" },
      timeScale: { borderColor: "#162038", fixLeftEdge: true, fixRightEdge: true },
      crosshair: {
        horzLine: { labelBackgroundColor: "#38bdf8" },
        vertLine: { labelBackgroundColor: "#38bdf8" },
      },
    });

    const series =
      mode === "candles"
        ? chart.addSeries(CandlestickSeries, {
            upColor: "#22c55e",
            downColor: "#ef4444",
            borderUpColor: "#22c55e",
            borderDownColor: "#ef4444",
            wickUpColor: "#22c55e",
            wickDownColor: "#ef4444",
            priceLineVisible: false,
          })
        : chart.addSeries(AreaSeries, {
            lineColor: "#38bdf8",
            lineWidth: 2,
            topColor: "rgba(56, 189, 248, 0.20)",
            bottomColor: "rgba(56, 189, 248, 0.01)",
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
  }, [mode]);

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
  }, [data, mode]);

  return <div ref={containerRef} className="w-full" />;
}

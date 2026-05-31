"use client";

import { useEffect, useRef } from "react";
import {
  AreaSeries,
  ColorType,
  createChart,
  type IChartApi,
  type ISeriesApi,
} from "lightweight-charts";
import type { PricePoint } from "@/lib/api";

export function PriceChart({ data }: { data: PricePoint[] }) {
  const containerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const seriesRef = useRef<ISeriesApi<"Area"> | null>(null);

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

    const series = chart.addSeries(AreaSeries, {
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
  }, []);

  useEffect(() => {
    if (!seriesRef.current) return;
    seriesRef.current.setData(
      data.map((p) => ({ time: p.date, value: p.close })),
    );
    chartRef.current?.timeScale().fitContent();
  }, [data]);

  return <div ref={containerRef} className="w-full" />;
}

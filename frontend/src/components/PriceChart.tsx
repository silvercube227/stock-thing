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

  // Create the chart once.
  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;

    const chart = createChart(el, {
      width: el.clientWidth,
      height: 300,
      layout: {
        background: { type: ColorType.Solid, color: "transparent" },
        textColor: "#8b8b94",
        fontFamily: "var(--font-geist-mono), monospace",
        attributionLogo: false,
      },
      grid: {
        vertLines: { visible: false },
        horzLines: { color: "#1a1a1f" },
      },
      rightPriceScale: { borderColor: "#232329" },
      timeScale: { borderColor: "#232329", fixLeftEdge: true, fixRightEdge: true },
      crosshair: { horzLine: { labelBackgroundColor: "#a78bfa" }, vertLine: { labelBackgroundColor: "#a78bfa" } },
    });

    const series = chart.addSeries(AreaSeries, {
      lineColor: "#a78bfa",
      lineWidth: 2,
      topColor: "rgba(167, 139, 250, 0.25)",
      bottomColor: "rgba(167, 139, 250, 0.02)",
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

  // Push data whenever it changes.
  useEffect(() => {
    if (!seriesRef.current) return;
    seriesRef.current.setData(
      data.map((p) => ({ time: p.date, value: p.close })),
    );
    chartRef.current?.timeScale().fitContent();
  }, [data]);

  return <div ref={containerRef} className="w-full" />;
}

"use client";

import { use, useEffect, useMemo } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { useAuth } from "@/components/AuthProvider";
import { AppHeader } from "@/components/AppHeader";
import { PriceChart } from "@/components/PriceChart";
import { RankGaugeRow } from "@/components/RankGauge";
import { FundamentalsPanel } from "@/components/FundamentalsPanel";
import { SentimentGauge } from "@/components/SentimentGauge";
import { useTickerDetail } from "@/hooks/useTickerDetail";
import { useQuotes } from "@/hooks/useQuotes";
import { changeColor, money, pct } from "@/lib/format";

function Card({
  title,
  children,
  className = "",
}: {
  title: string;
  children: React.ReactNode;
  className?: string;
}) {
  return (
    <section className={`rounded-2xl border border-border bg-surface p-5 ${className}`}>
      <h2 className="mb-4 text-xs font-medium text-muted">{title}</h2>
      {children}
    </section>
  );
}

export default function TickerPage({
  params,
}: {
  params: Promise<{ symbol: string }>;
}) {
  const { symbol } = use(params);
  const router = useRouter();
  const { session, loading: authLoading } = useAuth();
  const { detail, prices, loading, error } = useTickerDetail(symbol);

  useEffect(() => {
    if (!authLoading && !session) router.replace("/login");
  }, [authLoading, session, router]);

  const quoteSymbols = useMemo(() => [symbol], [symbol]);
  const quotes = useQuotes(quoteSymbols);
  const q = quotes[symbol.toUpperCase()];

  if (authLoading || !session) {
    return (
      <main className="flex flex-1 items-center justify-center text-sm text-muted">
        Loading…
      </main>
    );
  }

  const price = q?.price ?? detail?.last_close ?? null;

  return (
    <>
      <AppHeader>
        <span className="text-faint">/</span>
        <span className="text-sm font-semibold tracking-wide">{symbol.toUpperCase()}</span>
      </AppHeader>

      <main className="mx-auto w-full max-w-5xl flex-1 px-6 py-8">
        {error && (
          <p className="rounded-xl border border-down/30 bg-down/8 px-4 py-2.5 text-sm text-down mb-4">
            {error}
          </p>
        )}

        {loading && !detail ? (
          <div className="rounded-2xl border border-border p-12 text-center text-sm text-muted">
            Loading {symbol.toUpperCase()}…
          </div>
        ) : detail ? (
          <>
            <div className="mb-6 flex flex-wrap items-start justify-between gap-4">
              <div>
                <h1 className="text-2xl font-semibold tracking-tight">
                  {detail.ticker.symbol}
                </h1>
                <p className="mt-1 text-sm text-muted">
                  {detail.ticker.name}
                  {detail.ticker.sector ? (
                    <>
                      <span className="mx-1.5 text-faint">·</span>
                      <span className="text-faint">{detail.ticker.sector}</span>
                    </>
                  ) : null}
                </p>
              </div>
              <div className="text-right">
                <div className="nums text-2xl font-semibold">{money(price)}</div>
                {q?.change_pct != null && (
                  <div className={`nums text-sm font-medium mt-0.5 ${changeColor(q.change_pct)}`}>
                    {money(q.change)} ({pct(q.change_pct)})
                  </div>
                )}
              </div>
            </div>

            <div className="grid gap-4 lg:grid-cols-3">
              <Card title="Price history" className="lg:col-span-2">
                {prices.length > 0 ? (
                  <PriceChart data={prices} />
                ) : (
                  <p className="text-sm text-muted">No price history available.</p>
                )}
              </Card>

              <Card title="Projected performance">
                <RankGaugeRow
                  predictions={detail.predictions}
                  asOf={detail.as_of_date}
                />
              </Card>
            </div>

            <div className="mt-4 grid gap-4 lg:grid-cols-3">
              <Card title="Fundamentals" className="lg:col-span-2">
                <FundamentalsPanel f={detail.fundamentals} />
              </Card>
              <Card title="Sentiment">
                <SentimentGauge s={detail.sentiment} />
              </Card>
            </div>

            <div className="mt-6">
              <Link href="/" className="text-[11px] text-faint transition-colors hover:text-foreground">
                ← Back to portfolio
              </Link>
            </div>
          </>
        ) : null}
      </main>
    </>
  );
}

## Stock Trend Predictor

Personal long-only stock and ETF trend prediction app. Not for active trading — for directional awareness over 1M, 3M, 6M, and 1Y horizons.

### Stack
- **Frontend:** React (Vercel)
- **Backend:** FastAPI
- **Database:** Postgres (Supabase)
- **ML:** PyTorch LSTM, trained and served locally on M4 Mac via Metal backend
- **Sentiment:** FinBERT running locally, daily cron pushes scores to cloud Postgres

### Architecture
- Sentiment pipeline runs locally on a daily schedule, scores stored as 7-14 day rolling averages per ticker
- LSTM retrains automatically when new data arrives, triggered locally
- Model versions stored in DB with metadata (training date, data window, val loss, directional accuracy) — best performing version stays in production
- FastAPI handles inference requests, data queries, and orchestration
- React dashboard shows portfolio holdings, trend projections per ticker, and daily sentiment gauge

### Data Sources
- Price/volume: yfinance or Polygon.io
- Fundamentals: SEC EDGAR (10-K, 10-Q)
- Sentiment: Financial headlines via FinBERT

### Scope
- Options support is planned but deferred — American options pricing requires a separate model (Monte Carlo based)
- This is a personal tool, not a hosted service

That changes my answer a lot. S&P 500 from 2016-present including dropouts is a real dataset, not a toy one. It is enough that a transformer is no longer obviously doomed on sample size alone.
But it still needs a few things to be effective, and I think your current setup only has some of them.
The main missing piece is point-in-time universe membership. Your historical seeding script adds removed tickers with active=false, which is good, but the schema does not appear to track “was this stock actually in the S&P 500 on this sample date?” It has added_at and removed_at on tickers, but added_at defaults to insert time, not index-entry date. If you train on today’s S&P 500 names back to 2016 before they were actually members, there is still lookahead/survivorship contamination. For clean S&P 500 modeling, you want a table like:
sp500_membership(ticker_id, valid_from, valid_to, source)
Then every training cross-section should include only names that were index members on that date.
Second: your transformer is still mostly a per-ticker sequence model, not a true cross-sectional equity ranker. It gets one ticker’s 252 x 12 window plus a ticker embedding. It does not directly see the whole date’s cross-section the way LightGBM does after rank-normalization. For equity ranking, that matters. A better transformer would ingest per-date factor ranks or use cross-sectional attention over tickers.
Third: the input representation is probably still too raw. With 500-ish names, the model has more examples, but it still has to rediscover factors from sequence data. I’d want to feed it the same engineered features the GBDT gets: momentum ranks, volatility ranks, moving-average gaps, 52-week distance, liquidity, quality/fundamental ranks, sector-relative ranks, and market/sector context.
So the revised answer is:
Your dataset is probably large enough to justify a serious transformer experiment. But the current PatchTST is not yet the transformer I’d expect to win.
For a transformer to have a fair shot, I’d want:
* Point-in-time S&P 500 membership, not just current plus removed tickers.
* Monthly cross-sections built only from valid members on that date.
* Engineered factor sequences, preferably cross-sectionally rank-normalized.
* A ranking objective, not just return regression/classification.
* Horizon weights focused on 3M/6M/1Y, not 1M.
* Purged rolling validation and block-bootstrap significance.
* Seed ensembling, because one transformer run can easily be noise-selected.
* Ideally sector/industry embeddings and market-regime context.
My read: with your expanded S&P dataset, I would not permanently give up on transformers. But I would not expect the current raw-sequence PatchTST to beat LightGBM. The next fair transformer test should be “transformer over factor/rank panels with PIT membership,” not “same PatchTST, more tickers.”

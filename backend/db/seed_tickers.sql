-- Starter universe for stock-thing. Idempotent: ON CONFLICT DO NOTHING.
-- 4 broad ETFs + 31 large-cap equities (35 total), sector-diversified.
-- CIK is left NULL and populated by scripts/backfill_ciks.py.
--
-- Edit this list to fit your own watchlist BEFORE running. Once a ticker is
-- inserted it gets a permanent embedding_idx — removing it later means
-- setting active=false, not deleting the row (survivorship bias).

insert into tickers (symbol, name, asset_type, exchange, sector, industry) values
    -- ETFs
    ('SPY',   'SPDR S&P 500 ETF Trust',              'etf',    'NYSE Arca', 'Index',                'Broad Market'),
    ('QQQ',   'Invesco QQQ Trust',                   'etf',    'NASDAQ',    'Index',                'Large Cap Growth'),
    ('VTI',   'Vanguard Total Stock Market ETF',     'etf',    'NYSE Arca', 'Index',                'Broad Market'),
    ('DIA',   'SPDR Dow Jones Industrial Average',   'etf',    'NYSE Arca', 'Index',                'Large Cap Value'),

    -- Technology
    ('AAPL',  'Apple Inc.',                          'equity', 'NASDAQ',    'Technology',           'Consumer Electronics'),
    ('MSFT',  'Microsoft Corporation',               'equity', 'NASDAQ',    'Technology',           'Software'),
    ('GOOGL', 'Alphabet Inc. Class A',               'equity', 'NASDAQ',    'Communication Services','Interactive Media'),
    ('AMZN',  'Amazon.com Inc.',                     'equity', 'NASDAQ',    'Consumer Discretionary','Internet Retail'),
    ('NVDA',  'NVIDIA Corporation',                  'equity', 'NASDAQ',    'Technology',           'Semiconductors'),
    ('META',  'Meta Platforms Inc.',                 'equity', 'NASDAQ',    'Communication Services','Interactive Media'),
    ('TSLA',  'Tesla Inc.',                          'equity', 'NASDAQ',    'Consumer Discretionary','Auto Manufacturers'),
    ('AVGO',  'Broadcom Inc.',                       'equity', 'NASDAQ',    'Technology',           'Semiconductors'),
    ('ORCL',  'Oracle Corporation',                  'equity', 'NYSE',      'Technology',           'Software'),
    ('ADBE',  'Adobe Inc.',                          'equity', 'NASDAQ',    'Technology',           'Software'),

    -- Financials
    ('JPM',   'JPMorgan Chase & Co.',                'equity', 'NYSE',      'Financials',           'Diversified Banks'),
    ('BAC',   'Bank of America Corporation',         'equity', 'NYSE',      'Financials',           'Diversified Banks'),
    ('V',     'Visa Inc.',                           'equity', 'NYSE',      'Financials',           'Transaction Processing'),
    ('MA',    'Mastercard Incorporated',             'equity', 'NYSE',      'Financials',           'Transaction Processing'),
    ('WFC',   'Wells Fargo & Company',               'equity', 'NYSE',      'Financials',           'Diversified Banks'),

    -- Healthcare
    ('JNJ',   'Johnson & Johnson',                   'equity', 'NYSE',      'Healthcare',           'Pharmaceuticals'),
    ('UNH',   'UnitedHealth Group Inc.',             'equity', 'NYSE',      'Healthcare',           'Managed Healthcare'),
    ('LLY',   'Eli Lilly and Company',               'equity', 'NYSE',      'Healthcare',           'Pharmaceuticals'),
    ('PFE',   'Pfizer Inc.',                         'equity', 'NYSE',      'Healthcare',           'Pharmaceuticals'),

    -- Consumer Discretionary / Staples
    ('WMT',   'Walmart Inc.',                        'equity', 'NYSE',      'Consumer Staples',     'Hypermarkets'),
    ('COST',  'Costco Wholesale Corporation',        'equity', 'NASDAQ',    'Consumer Staples',     'Hypermarkets'),
    ('HD',    'The Home Depot Inc.',                 'equity', 'NYSE',      'Consumer Discretionary','Home Improvement'),
    ('NKE',   'NIKE Inc.',                           'equity', 'NYSE',      'Consumer Discretionary','Footwear & Apparel'),
    ('MCD',   'McDonald''s Corporation',             'equity', 'NYSE',      'Consumer Discretionary','Restaurants'),
    ('KO',    'The Coca-Cola Company',               'equity', 'NYSE',      'Consumer Staples',     'Soft Drinks'),
    ('PEP',   'PepsiCo Inc.',                        'equity', 'NASDAQ',    'Consumer Staples',     'Soft Drinks'),
    ('PG',    'The Procter & Gamble Company',        'equity', 'NYSE',      'Consumer Staples',     'Household Products'),

    -- Energy / Industrials
    ('XOM',   'Exxon Mobil Corporation',             'equity', 'NYSE',      'Energy',               'Integrated Oil & Gas'),
    ('CVX',   'Chevron Corporation',                 'equity', 'NYSE',      'Energy',               'Integrated Oil & Gas'),
    ('BA',    'The Boeing Company',                  'equity', 'NYSE',      'Industrials',          'Aerospace & Defense'),
    ('CAT',   'Caterpillar Inc.',                    'equity', 'NYSE',      'Industrials',          'Construction Machinery')
on conflict do nothing;

-- Sanity check after running:
--   select count(*) from tickers where active = true;       -- expect 35
--   select symbol, sector, embedding_idx from tickers order by embedding_idx;

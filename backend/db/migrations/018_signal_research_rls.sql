-- Research inputs are backend-only. No browser/API role policies are granted.
alter table security_identifiers enable row level security;
alter table security_events enable row level security;
alter table accounting_facts enable row level security;
alter table fixed_estimates enable row level security;
alter table macro_vintages enable row level security;
alter table research_news_daily enable row level security;
alter table research_news_coverage enable row level security;

-- RLS covers row access; revoke grants too (including non-row privileges).
revoke all on table security_identifiers, security_events, accounting_facts,
    fixed_estimates, macro_vintages, research_news_daily, research_news_coverage
    from public, anon, authenticated;

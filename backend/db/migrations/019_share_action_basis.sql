-- Price adjustment factors can include spin-offs and are NOT share-count ratios.
-- Unknown event classifications remain NULL; never copy split_factor wholesale.
alter table price_history add column if not exists share_split_factor numeric;
alter table price_history add column if not exists share_action_source text;
comment on column price_history.share_split_factor is
    'Verified new/old shares of this security; pricing-only distributions use 1.';
comment on column price_history.split_factor is
    'Source price adjustment ratio, potentially including pricing-only distributions.';

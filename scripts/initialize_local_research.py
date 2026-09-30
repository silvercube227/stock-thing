"""Initialize only the dedicated local research database, never a configured remote DB."""

import argparse
import asyncio
from pathlib import Path
from urllib.parse import urlparse

import asyncpg
from dotenv import dotenv_values

from backend.ml.research import write_json_new
from scripts.apply_registered_membership import digest


async def run(config, output):
    settings = dotenv_values(config)
    dsn = settings["DATABASE_URL"]
    url = urlparse(dsn)
    if (
        url.hostname != "127.0.0.1"
        or url.port != 55432
        or url.path != "/stock_research"
        or url.username != "research_admin"
    ):
        raise ValueError("Only the dedicated local research database is allowed")
    if settings.get("FRAME_CACHE_DIR") != ".frame_cache_research/local-sp1500":
        raise ValueError("Dedicated frame cache required")
    files = [Path("backend/db/schema.sql"), *sorted(Path("backend/db/migrations").glob("*.sql"))]
    conn = await asyncpg.connect(dsn)
    try:
        async with conn.transaction():
            if await conn.fetchval(
                "select count(*) from information_schema.tables where table_schema='public'"
            ):
                raise ValueError("Initialization requires an empty local database")
            await conn.execute("""create role anon nologin; create role authenticated nologin;
                create schema auth; create table auth.users(id uuid primary key);
                comment on table auth.users is 'Empty local compatibility table; no users.';""")
            for path in files:
                await conn.execute(path.read_text())
            tables = [
                r["table_name"]
                for r in await conn.fetch("""select table_name from
                information_schema.tables where table_schema='public' order by table_name""")
            ]
            identities = await conn.fetchval("select count(*) from tickers")
            assert identities == 0
        result = dict(
            status="local_schema_initialized_universe_data_pending",
            database="stock_research",
            host="127.0.0.1",
            port=55432,
            version=await conn.fetchval("show server_version"),
            tables=tables,
            source_hashes={str(p): digest(p.read_bytes()) for p in files},
            frame_cache=settings["FRAME_CACHE_DIR"],
            news_cache=settings["NEWS_CACHE_DIR"],
            production_data_copied=False,
            research_securities=identities,
        )
        write_json_new(output, result)
        print({k: v for k, v in result.items() if k not in ("source_hashes", "tables")})
    finally:
        await conn.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default=".research/local-postgres/research.env")
    p.add_argument("--output", required=True)
    a = p.parse_args()
    asyncio.run(run(a.config, a.output))

"""Promote a `candidate` model_version to `production`.

What it does, atomically inside one transaction:
  1. Resolves a target candidate (latest by created_at, or --model-version-id).
  2. Marks the current production row (if any) as `retired`, stamping
     `retired_at = now()`.
  3. Updates the target row to `status = 'production'` and stamps `promoted_at`.

The `model_versions_one_production` partial unique index in schema.sql means a
non-transactional sequence (set new = production, then retire old) would briefly
violate uniqueness — so we do it the other way: retire old first, promote new
second, both inside one txn. The unique index is then never observed to hold
more than one production row.

This script does NOT touch the `predictions` table — the candidate's predictions
were already written by `gbm_inference.py` and remain valid; the dashboard's
`_resolve_active_model` reads `predictions` keyed by the now-production
`model_version_id`.

Usage:
  python -m backend.jobs.promote_model                       # latest candidate
  python -m backend.jobs.promote_model --model-version-id X  # specific row
  python -m backend.jobs.promote_model --dry-run             # show intent only
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from backend.ingestion.db import pool_context


async def _resolve_candidate(pool, model_version_id: str | None):
    """Return the candidate row to promote, or raise SystemExit with context."""
    if model_version_id is not None:
        row = await pool.fetchrow(
            "select model_version_id, status, created_at, config"
            "  from model_versions"
            " where model_version_id = $1",
            model_version_id,
        )
        if row is None:
            raise SystemExit(f"no model_version found with id={model_version_id}")
        if row["status"] != "candidate":
            raise SystemExit(
                f"model_version {model_version_id} has status={row['status']}, "
                f"not 'candidate' — refusing to promote (use --force to override)"
            )
        return row

    row = await pool.fetchrow(
        "select model_version_id, status, created_at, config"
        "  from model_versions"
        " where status = 'candidate'"
        " order by created_at desc"
        " limit 1"
    )
    if row is None:
        raise SystemExit("no candidate model_versions found")
    return row


def _summarize_specs(config: dict | None) -> str:
    """One-line summary of per-horizon training targets for the log message."""
    if not config:
        return "(no config)"
    if "specs" in config and isinstance(config["specs"], dict):
        parts = [
            f"{h}={spec.get('target_mode', '?')}"
            for h, spec in config["specs"].items()
        ]
        return ", ".join(parts)
    return f"(legacy cfg, target_mode={config.get('target_mode', '?')})"


async def run(args) -> None:
    import json

    async with pool_context() as pool:
        cand = await _resolve_candidate(pool, args.model_version_id)
        cand_cfg = json.loads(cand["config"]) if isinstance(cand["config"], str) else cand["config"]
        cand_id = str(cand["model_version_id"])

        current = await pool.fetchrow(
            "select model_version_id, created_at, promoted_at, config"
            "  from model_versions"
            " where status = 'production'"
            " limit 1"
        )

        print(f"target candidate: {cand_id}")
        print(f"  created_at: {cand['created_at']}")
        print(f"  targets:    {_summarize_specs(cand_cfg)}")
        if current is not None:
            cur_cfg = json.loads(current["config"]) if isinstance(current["config"], str) else current["config"]
            print(f"current production: {current['model_version_id']}")
            print(f"  promoted_at: {current['promoted_at']}")
            print(f"  targets:     {_summarize_specs(cur_cfg)}")
        else:
            print("current production: (none)")

        if args.dry_run:
            print("--dry-run: not applying changes")
            return

        async with pool.acquire() as conn:
            async with conn.transaction():
                if current is not None:
                    await conn.execute(
                        "update model_versions set status='retired', retired_at=now() "
                        "where model_version_id = $1",
                        current["model_version_id"],
                    )
                await conn.execute(
                    "update model_versions set status='production', promoted_at=now() "
                    "where model_version_id = $1",
                    cand["model_version_id"],
                )
        print(f"promoted {cand_id} to production"
              + (f"; retired {current['model_version_id']}" if current is not None else ""))


def main() -> None:
    p = argparse.ArgumentParser(
        description="Promote a candidate GBDT model_version to production."
    )
    p.add_argument(
        "--model-version-id",
        help="specific candidate UUID to promote (default: latest by created_at)",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="print what would happen; don't write to model_versions",
    )
    args = p.parse_args()
    try:
        asyncio.run(run(args))
    except SystemExit as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

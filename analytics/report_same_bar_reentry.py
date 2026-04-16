#!/usr/bin/env python3
"""
Отчёт по SAME_BAR_REENTRY (same-bar guard) в decision_logs.

Запуск из корня проекта:
  ANALYTICS_DB_PATH=analytics.db python3 -m analytics.report_same_bar_reentry
  python3 -m analytics.report_same_bar_reentry /path/to/analytics.db
"""
from __future__ import annotations

import os
import sqlite3
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _db_path() -> Path:
    raw = os.getenv("ANALYTICS_DB_PATH", "analytics.db").strip()
    p = Path(raw)
    return p if p.is_absolute() else (_ROOT / p)


def main() -> None:
    db = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else _db_path()
    if not db.exists():
        print(f"DB not found: {db}", file=sys.stderr)
        sys.exit(1)
    conn = sqlite3.connect(str(db))
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    cur.execute(
        """
        SELECT COUNT(*) AS n
        FROM decision_logs
        WHERE reason_code = 'SAME_BAR_REENTRY'
        """
    )
    total = int(cur.fetchone()["n"])
    print("=== SAME_BAR_REENTRY (total) ===")
    print(total)

    print("\n=== By ticker ===")
    cur.execute(
        """
        SELECT ticker, COUNT(*) AS cnt
        FROM decision_logs
        WHERE reason_code = 'SAME_BAR_REENTRY'
        GROUP BY ticker
        ORDER BY cnt DESC, ticker ASC
        """
    )
    for row in cur.fetchall():
        print(f"  {row['ticker']}: {row['cnt']}")

    print("\n=== By hour (UTC, from decision_ts) ===")
    cur.execute(
        """
        SELECT
          CASE
            WHEN instr(decision_ts, 'T') > 0
            THEN substr(decision_ts, instr(decision_ts, 'T') + 1, 2)
            ELSE '??'
          END AS hour_utc,
          COUNT(*) AS cnt
        FROM decision_logs
        WHERE reason_code = 'SAME_BAR_REENTRY'
        GROUP BY hour_utc
        ORDER BY hour_utc
        """
    )
    for row in cur.fetchall():
        print(f"  {row['hour_utc']}:00 UTC — {row['cnt']}")

    conn.close()


if __name__ == "__main__":
    main()

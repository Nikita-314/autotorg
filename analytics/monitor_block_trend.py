#!/usr/bin/env python3
"""
Мониторинг накопления outcomes для BLOCK_TREND (только чтение БД + опционально batch evaluate_pending).
Не меняет стратегию и bot.py.

Запуск из корня проекта:
  python3 -m analytics.monitor_block_trend
  python3 -m analytics.monitor_block_trend --batch-eval
"""
from __future__ import annotations

import os
import sqlite3
import statistics
import sys
import time
from pathlib import Path

# Пакет analytics при запуске как скрипта: добавляем корень проекта в path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _db_path() -> Path:
    root = Path(__file__).resolve().parent.parent
    raw = os.getenv("ANALYTICS_DB_PATH", "analytics.db").strip()
    p = Path(raw)
    return p if p.is_absolute() else (root / p)


def fetch_block_trend_stats(conn: sqlite3.Connection) -> dict:
    cur = conn.cursor()
    cur.execute(
        """
        SELECT o.outcome_15m_pct
        FROM signals s
        JOIN signal_outcomes o ON o.signal_id = s.signal_id
        WHERE s.reason_code = 'BLOCK_TREND'
          AND o.outcome_15m_pct IS NOT NULL
        """
    )
    vals = [r[0] for r in cur.fetchall()]
    n = len(vals)
    if n == 0:
        return {
            "count": 0,
            "avg_15m": None,
            "median_15m": None,
            "pct_pos": None,
            "pct_neg": None,
            "pct_zero": None,
        }
    pos = sum(1 for v in vals if v > 0)
    neg = sum(1 for v in vals if v < 0)
    zer = sum(1 for v in vals if v == 0)
    return {
        "count": n,
        "avg_15m": sum(vals) / n,
        "median_15m": statistics.median(vals),
        "pct_pos": 100.0 * pos / n,
        "pct_neg": 100.0 * neg / n,
        "pct_zero": 100.0 * zer / n,
    }


def print_report(stats: dict) -> None:
    print("=== BLOCK_TREND outcomes (outcome_15m_pct NOT NULL) ===")
    print(f"1. count:              {stats['count']}")
    print(f"2. avg outcome_15m:   {stats['avg_15m']}")
    print(f"3. median outcome_15m: {stats['median_15m']}")
    print(f"4. % positive:         {stats['pct_pos']}")
    print(f"5. % negative:         {stats['pct_neg']}")
    print(f"6. % zero:             {stats['pct_zero']}")
    if stats["count"] < 200:
        print()
        print("DATA TOO SMALL (need >= 200 rows for stable monitoring)")
    else:
        print()
        print("DATA OK (count >= 200)")


def run_batch_evaluate_pending() -> None:
    """10x: evaluate_pending(300), пауза 60 с. Требует venv и data_loader."""
    from analytics.db import AnalyticsDB
    from analytics.outcome_evaluator import OutcomeEvaluator
    from analytics.signal_logger import SignalLogger
    from data_loader import load_candles

    db_path = _db_path()
    exchange = os.getenv("TV_EXCHANGE", "MOEX").strip()
    interval = os.getenv("TV_INTERVAL", "1h").strip()

    def load_fn(ticker: str):
        return load_candles(ticker, exchange, interval)

    before = sqlite3.connect(str(db_path))
    s0 = fetch_block_trend_stats(before)
    before.close()

    db = AnalyticsDB(db_path)
    ev = OutcomeEvaluator(db, SignalLogger(db))

    print("=== BATCH: evaluate_pending(limit=300) x10, sleep 60s ===")
    print(f"DB: {db_path}")
    print(f"BLOCK_TREND count before: {s0['count']}")
    print()

    for i in range(1, 11):
        updated = ev.evaluate_pending(load_fn, limit=300, reeval_null_older_than_minutes=0)
        print(f"[{i}/10] updated={updated}")
        time.sleep(60)

    db.conn.close()

    after = sqlite3.connect(str(db_path))
    s1 = fetch_block_trend_stats(after)
    after.close()

    print()
    print("=== AFTER BATCH ===")
    print(f"BLOCK_TREND outcomes count: {s1['count']} (delta +{s1['count'] - s0['count']})")
    print_report(s1)


def main() -> None:
    if "--batch-eval" in sys.argv:
        run_batch_evaluate_pending()
        return

    db_path = _db_path()
    if not db_path.is_file():
        print(f"DB not found: {db_path}", file=sys.stderr)
        sys.exit(1)

    conn = sqlite3.connect(str(db_path))
    try:
        stats = fetch_block_trend_stats(conn)
        print_report(stats)
    finally:
        conn.close()


if __name__ == "__main__":
    main()

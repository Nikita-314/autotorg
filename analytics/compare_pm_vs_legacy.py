#!/usr/bin/env python3
"""
Сравнение PM (shadow) vs legacy по analytics.db.

Запуск из корня проекта:
  ANALYTICS_DB_PATH=analytics.db python3 -m analytics.compare_pm_vs_legacy
  python3 -m analytics.compare_pm_vs_legacy --since 2026-03-26T00:00:00

В shadow режиме label PARTIAL_EXIT не пишется (только paper); частичный сценарий
смотрите по SOFT_STOP_HIT + details.close_fraction > 0.
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _db_path() -> Path:
    raw = os.getenv("ANALYTICS_DB_PATH", "analytics.db").strip()
    p = Path(raw)
    return p if p.is_absolute() else (_ROOT / p)


PM_CODES = [
    "POSITION_STATE_INIT",
    "SOFT_STOP_HIT",
    "HARD_STOP_HIT",
    "TP_ZONE_ENTERED",
    "TRAILING_ACTIVATED",
    "TRAILING_UPDATED",
    "TRAILING_EXIT",
]


def load_rows(conn: sqlite3.Connection, since: Optional[str]) -> Tuple[List[dict], List[dict]]:
    cur = conn.cursor()
    if since:
        pm_sql = """
            SELECT decision_id, ticker, decision_ts, decision_type, decision_label, reason_code, reason_text, details_json, signal_id
            FROM decision_logs
            WHERE decision_type = 'POSITION_MGMT' AND decision_ts >= ?
            ORDER BY decision_ts ASC
        """
        leg_sql = """
            SELECT decision_id, ticker, decision_ts, decision_type, decision_label, reason_code, reason_text, details_json, signal_id
            FROM decision_logs
            WHERE decision_type = 'TRADE_CLOSE' AND decision_ts >= ?
            ORDER BY decision_ts ASC
        """
        cur.execute(pm_sql, (since,))
        pm_rows = [dict(zip([c[0] for c in cur.description], r)) for r in cur.fetchall()]
        cur.execute(leg_sql, (since,))
        leg_rows = [dict(zip([c[0] for c in cur.description], r)) for r in cur.fetchall()]
    else:
        cur.execute(
            """
            SELECT decision_id, ticker, decision_ts, decision_type, decision_label, reason_code, reason_text, details_json, signal_id
            FROM decision_logs
            WHERE decision_type = 'POSITION_MGMT'
            ORDER BY decision_ts ASC
            """
        )
        pm_rows = [dict(zip([c[0] for c in cur.description], r)) for r in cur.fetchall()]
        cur.execute(
            """
            SELECT decision_id, ticker, decision_ts, decision_type, decision_label, reason_code, reason_text, details_json, signal_id
            FROM decision_logs
            WHERE decision_type = 'TRADE_CLOSE'
            ORDER BY decision_ts ASC
            """
        )
        leg_rows = [dict(zip([c[0] for c in cur.description], r)) for r in cur.fetchall()]
    return pm_rows, leg_rows


def details(obj: dict) -> dict:
    try:
        return json.loads(obj.get("details_json") or "{}")
    except json.JSONDecodeError:
        return {}


def main() -> None:
    ap = argparse.ArgumentParser(description="PM vs legacy comparison from decision_logs")
    ap.add_argument("--since", type=str, default=None, help="ISO prefix filter, e.g. 2026-03-26T00:00:00")
    ap.add_argument("--db", type=str, default=None, help="Path to analytics.db")
    args = ap.parse_args()
    db_path = Path(args.db) if args.db else _db_path()
    if not db_path.exists():
        print(f"DB not found: {db_path}", file=sys.stderr)
        sys.exit(1)

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    pm_rows, leg_rows = load_rows(conn, args.since)
    conn.close()

    pm_by_code = Counter()
    partial_proxy = 0
    for r in pm_rows:
        code = r.get("reason_code") or r.get("decision_label") or ""
        pm_by_code[code] += 1
        d = details(r)
        if code == "SOFT_STOP_HIT" and float(d.get("close_fraction") or 0) > 0:
            partial_proxy += 1

    leg_count = len(leg_rows)
    legacy_reasons = Counter(r.get("reason_text") or "" for r in leg_rows)

    print("=== 1. Сводка ===")
    print(f"TRADE_CLOSE (legacy, всего в периоде): {leg_count}")
    print("  по reason_text:", dict(legacy_reasons))
    print()
    print("PM события (decision_type=POSITION_MGMT, reason_code):")
    for c in PM_CODES + ["POSITION_STATE_INIT"]:
        if c in pm_by_code:
            print(f"  {c}: {pm_by_code[c]}")
    for k, v in sorted(pm_by_code.items()):
        if k not in PM_CODES and k != "POSITION_STATE_INIT":
            print(f"  {k}: {v}")
    print()
    print(f"SOFT_STOP_HIT с close_fraction>0 (прокси partial в shadow): {partial_proxy}")
    print(f"PARTIAL_EXIT (label) — только в paper; в shadow: {sum(1 for r in pm_rows if r.get('decision_label')=='PARTIAL_EXIT')}")

    # Per signal_id: PM exit vs legacy
    pm_by_signal: Dict[str, List[dict]] = defaultdict(list)
    for r in pm_rows:
        sid = r.get("signal_id")
        if sid:
            pm_by_signal[sid].append(r)

    leg_by_signal: Dict[str, dict] = {}
    for r in leg_rows:
        sid = r.get("signal_id")
        if sid:
            leg_by_signal[sid] = r

    pm_full_exit_codes = {"HARD_STOP_HIT", "TRAILING_EXIT"}

    earlier = later = same_ts = 0
    comparable = 0
    pm_trailing_exit_count = sum(1 for r in pm_rows if (r.get("reason_code") == "TRAILING_EXIT"))
    pm_partial_signals = 0

    print("\n=== 2. По signal_id (есть TRADE_CLOSE и хотя бы одно PM событие) ===")
    for sid, close_row in leg_by_signal.items():
        if sid not in pm_by_signal:
            continue
        evs = sorted(pm_by_signal[sid], key=lambda x: x["decision_ts"])
        close_ts = close_row["decision_ts"]
        legacy_reason = close_row.get("reason_text") or ""

        pm_full_ts = None
        pm_stage: Optional[str] = None
        would_partial = any(
            (e.get("reason_code") == "SOFT_STOP_HIT" and float(details(e).get("close_fraction") or 0) > 0)
            for e in evs
        )
        if would_partial:
            pm_partial_signals += 1

        for ev in evs:
            code = ev.get("reason_code") or ""
            if code in pm_full_exit_codes:
                pm_full_ts = ev["decision_ts"]
                pm_stage = code
                break

        exit_kind = "full" if pm_full_ts else ("partial" if would_partial else "no_pm_exit")

        if pm_full_ts:
            comparable += 1
            if pm_full_ts < close_ts:
                earlier += 1
            elif pm_full_ts > close_ts:
                later += 1
            else:
                same_ts += 1

        print(
            f"  {close_row['ticker']} signal={sid[:8]}... "
            f"legacy={legacy_reason!r} pm_full_stage={pm_stage!r} "
            f"would={exit_kind}"
        )

    print("\n=== 3. Метрики PM vs legacy (full exit PM vs время TRADE_CLOSE) ===")
    print(f"  пар с PM full-exit событием: {comparable}")
    print(f"  PM full exit раньше TRADE_CLOSE: {earlier}")
    print(f"  PM full exit позже TRADE_CLOSE: {later}")
    print(f"  одинаковый ts: {same_ts}")
    print(f"  сигналов с хотя бы одним SOFT partial (прокси): {pm_partial_signals}")
    print(f"  всего PM TRAILING_EXIT строк: {pm_trailing_exit_count}")
    print(
        "\n  Пояснение: в shadow legacy реально закрывает позицию; "
        "ts PM — момент, когда evaluate_tick сгенерировал бы полный выход."
    )

    print("\n=== 4. По тикерам: последние legacy причины ===")
    by_ticker: Dict[str, List[str]] = defaultdict(list)
    for r in leg_rows:
        by_ticker[r["ticker"]].append(r.get("reason_text") or "")
    for t in sorted(by_ticker.keys()):
        print(f"  {t}: {Counter(by_ticker[t])}")


if __name__ == "__main__":
    main()

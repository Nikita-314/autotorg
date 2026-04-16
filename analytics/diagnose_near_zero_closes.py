#!/usr/bin/env python3
"""
Диагностика почти нулевых закрытий (только чтение БД / paper_state).
Запуск: python3 -m analytics.diagnose_near_zero_closes
"""
from __future__ import annotations

import json
import os
import sqlite3
import sys
from collections import Counter
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _db_path() -> Path:
    raw = os.getenv("ANALYTICS_DB_PATH", "analytics.db").strip()
    p = Path(raw)
    return p if p.is_absolute() else (_ROOT / p)


def load_closes_from_decision_logs(conn: sqlite3.Connection) -> list[dict]:
    cur = conn.cursor()
    cur.execute(
        """
        SELECT ticker, decision_ts, details_json, reason_text
        FROM decision_logs
        WHERE decision_type = 'TRADE_CLOSE'
        ORDER BY decision_ts ASC
        """
    )
    out = []
    for ticker, ts, dj, rtext in cur.fetchall():
        row = {}
        if dj:
            try:
                row = json.loads(dj)
            except json.JSONDecodeError:
                row = {}
        row["_ticker"] = ticker
        row["_ts"] = ts
        row["_reason"] = rtext
        out.append(row)
    return out


def paper_links_fifo(conn: sqlite3.Connection) -> list[dict]:
    """Пары OPEN→CLOSE по тикеру (FIFO); pnl ≈ (exit-open)*qty по ценам линков."""
    cur = conn.cursor()
    cur.execute(
        """
        SELECT ticker, side, event_ts, price, qty, comment
        FROM paper_trade_links
        ORDER BY event_ts ASC
        """
    )
    rows = cur.fetchall()
    by_t: dict[str, list] = {}
    for r in rows:
        by_t.setdefault(r[0], []).append(r)

    pairs = []
    for ticker, evs in by_t.items():
        q: list = []
        for trow in evs:
            tkr, side, ts, price, qty, comment = trow
            if side == "OPEN":
                q.append((ts, price, qty, comment))
            elif side == "CLOSE" and q:
                ots, op, oqty, oc = q.pop(0)
                cpx, cqty, cc = price, qty, comment
                # long: pnl rub ≈ (exit - entry) * qty_close (используем qty из CLOSE)
                pnl_est = (float(cpx) - float(op)) * float(cqty)
                pairs.append(
                    {
                        "ticker": ticker,
                        "open_ts": ots,
                        "close_ts": ts,
                        "entry_px": float(op),
                        "exit_px": float(cpx),
                        "qty": float(cqty),
                        "pnl_est_rub": pnl_est,
                        "same_px": abs(float(cpx) - float(op)) < 1e-9,
                        "comment": cc,
                    }
                )
    return pairs


def main() -> None:
    db_path = _db_path()
    paper_state = _ROOT / os.getenv("PAPER_STATE_FILE", "paper_state.json")

    print("=== Источники ===")
    print(f"decision_logs / DB: {db_path}")
    print(f"paper_state.json:   {paper_state}")

    if not db_path.is_file():
        print(f"DB not found: {db_path}", file=sys.stderr)
        sys.exit(1)

    conn = sqlite3.connect(str(db_path))
    try:
        closes = load_closes_from_decision_logs(conn)
    finally:
        conn.close()

    n = len(closes)
    lt1 = sum(1 for r in closes if r.get("pnl") is not None and abs(float(r["pnl"])) < 1)
    lt10 = sum(1 for r in closes if r.get("pnl") is not None and abs(float(r["pnl"])) < 10)
    pct005 = sum(
        1
        for r in closes
        if r.get("pnl_pct") is not None and abs(float(r["pnl_pct"])) < 0.05
    )
    same_px = sum(
        1
        for r in closes
        if r.get("avg_price") is not None
        and r.get("exit_price") is not None
        and abs(float(r["avg_price"]) - float(r["exit_price"])) < 1e-9
    )

    print("\n=== decision_logs (TRADE_CLOSE) ===")
    print(f"1. Всего закрытых сделок:     {n}")
    print(f"2. |pnl| < 1 ₽:               {lt1}")
    print(f"3. |pnl| < 10 ₽:              {lt10}")
    print(f"4. |pnl_pct| < 0.05%:         {pct005}")
    print(f"5. avg_price == exit_price:  {same_px}")

    # Почти нулевые: |pnl|<10 или |pnl_pct|<0.05
    def is_near_zero(r: dict) -> bool:
        p = r.get("pnl")
        pc = r.get("pnl_pct")
        if p is not None and abs(float(p)) < 10:
            return True
        if pc is not None and abs(float(pc)) < 0.05:
            return True
        return False

    nz = [r for r in closes if is_near_zero(r)]
    c = Counter(str(r["_ticker"]) for r in nz)
    print("\n6. Топ тикеров по «почти нулевым» (|pnl|<10 ИЛИ |pnl_pct|<0.05%):")
    for t, k in c.most_common(15):
        print(f"   {t}: {k}")

    # paper_trade_links
    conn = sqlite3.connect(str(db_path))
    try:
        pairs = paper_links_fifo(conn)
    finally:
        conn.close()

    print("\n=== paper_trade_links (оценка по ценам OPEN/CLOSE) ===")
    print(f"Сопоставленных пар OPEN→CLOSE: {len(pairs)}")
    nz_pl = [p for p in pairs if abs(p["pnl_est_rub"]) < 10]
    same_pl = sum(1 for p in pairs if p["same_px"])
    print(f"|pnl_est| < 10 ₽: {len(nz_pl)}")
    print(f"entry_px == exit_px: {same_pl}")

    c2 = Counter(p["ticker"] for p in pairs if abs(p["pnl_est_rub"]) < 10 or p["same_px"])
    print("Топ тикеров (|pnl_est|<10 или same_px):")
    for t, k in c2.most_common(10):
        print(f"   {t}: {k}")

    if paper_state.is_file():
        with paper_state.open("r", encoding="utf-8") as f:
            st = json.load(f)
        print("\n=== paper_state.json (агрегаты) ===")
        print(f"trade_count: {st.get('trade_count')}")
        print(f"realized_pnl_rub: {st.get('realized_pnl_rub')}")
        print("(поштучных закрытий в файле нет — только счётчики)")


if __name__ == "__main__":
    main()

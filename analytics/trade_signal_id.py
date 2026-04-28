"""Resolve BUY signal_id for TRADE_CLOSE when in-memory trade_meta is empty (e.g. after restart)."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from .db import AnalyticsDB


def resolve_buy_signal_id_at_close(
    db: Optional[AnalyticsDB],
    ticker: str,
    trade_meta_entry: Optional[Dict[str, Any]],
) -> Tuple[Optional[str], Optional[str]]:
    """
    Return (signal_id, signal_id_missing_reason).
    Missing reason is set only when signal_id is None after resolution attempts.
    """
    if trade_meta_entry:
        sid = trade_meta_entry.get("signal_id")
        if sid:
            return str(sid), None

    if db is None:
        if trade_meta_entry is None:
            return None, "trade_meta_missing_after_restart"
        return None, "analytics_db_unavailable"

    row = db.fetchone(
        """
        SELECT d.signal_id
        FROM decision_logs d
        WHERE d.ticker = ? AND d.decision_type = 'TRADE_OPEN' AND d.signal_id IS NOT NULL
          AND NOT EXISTS (
            SELECT 1 FROM decision_logs c
            WHERE c.signal_id = d.signal_id AND c.decision_type = 'TRADE_CLOSE'
          )
        ORDER BY d.decision_ts DESC
        LIMIT 1
        """,
        (ticker,),
    )
    if row and row["signal_id"]:
        return str(row["signal_id"]), None

    row2 = db.fetchone(
        """
        SELECT p.signal_id
        FROM paper_trade_links p
        WHERE p.ticker = ? AND p.side = 'OPEN' AND p.signal_id IS NOT NULL
          AND NOT EXISTS (
            SELECT 1 FROM decision_logs c
            WHERE c.signal_id = p.signal_id AND c.decision_type = 'TRADE_CLOSE'
          )
        ORDER BY p.event_ts DESC
        LIMIT 1
        """,
        (ticker,),
    )
    if row2 and row2["signal_id"]:
        return str(row2["signal_id"]), None

    if trade_meta_entry is None:
        return None, "trade_meta_missing_after_restart"
    if not trade_meta_entry.get("signal_id"):
        return None, "signal_id_missing_in_trade_meta_unrecoverable"
    return None, "signal_id_unrecoverable_no_matching_open_record"

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from analytics.db import AnalyticsDB

from .buckets import BucketKey, volume_ratio_to_bucket


@dataclass
class BucketTradeStats:
    bucket: BucketKey
    n_signals: int
    sum_net_pnl: float
    avg_net_pnl: float
    avg_net_pnl_pct: float
    sum_gross: float
    gross_pos_net_nonpos: int
    near_zero_churn: int


@dataclass
class PaperBalanceSnapshot:
    initial_balance_rub: float
    current_balance_rub: float
    realized_gross_pnl_rub: float
    realized_commission_rub: float
    realized_net_pnl_rub: float
    unrealized_pnl_rub: float
    equity_rub: float
    last_reset_at: Optional[str]


def load_paper_balance(path: Path) -> Optional[PaperBalanceSnapshot]:
    if not path.exists():
        return None
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
        return PaperBalanceSnapshot(
            initial_balance_rub=float(d.get("initial_balance_rub", 0)),
            current_balance_rub=float(d.get("current_balance_rub", 0)),
            realized_gross_pnl_rub=float(d.get("realized_gross_pnl_rub", 0)),
            realized_commission_rub=float(d.get("realized_commission_rub", 0)),
            realized_net_pnl_rub=float(d.get("realized_net_pnl_rub", 0)),
            unrealized_pnl_rub=float(d.get("unrealized_pnl_rub", 0)),
            equity_rub=float(d.get("equity_rub", d.get("current_balance_rub", 0))),
            last_reset_at=d.get("last_reset_at"),
        )
    except Exception:
        return None


def _vr_expr(alias: str = "s") -> str:
    return f"""COALESCE(
        json_extract({alias}.feature_snapshot_json, '$.volume_ratio'),
        json_extract({alias}.feature_snapshot_json, '$.vol_ratio')
    )"""


def fetch_bucket_close_stats(db: AnalyticsDB, observation_days: int) -> Dict[BucketKey, BucketTradeStats]:
    mod = f"-{int(observation_days)} days"
    vr = _vr_expr("s")
    sql = f"""
    SELECT s.signal_id,
           MAX(CAST({vr} AS REAL)) AS vr,
           SUM(json_extract(c.details_json, '$.net_pnl')) AS sum_net,
           AVG(json_extract(c.details_json, '$.net_pnl_pct')) AS avg_net_pct,
           SUM(json_extract(c.details_json, '$.gross_pnl')) AS sum_gross,
           COUNT(*) AS n_closes
    FROM signals s
    JOIN decision_logs c
      ON c.signal_id = s.signal_id AND c.decision_type = 'TRADE_CLOSE'
    WHERE s.side = 'BUY'
      AND datetime(s.signal_ts) >= datetime('now', ?)
    GROUP BY s.signal_id
    HAVING vr IS NOT NULL
    """
    rows = db.fetchall(sql, (mod,))
    accum: Dict[BucketKey, List[Dict[str, Any]]] = {"lt1": [], "b12": [], "gt2": []}
    for r in rows:
        vr_val = float(r["vr"])
        bk = volume_ratio_to_bucket(vr_val)
        if bk is None:
            continue
        gross = float(r["sum_gross"] or 0)
        net = float(r["sum_net"] or 0)
        row = {
            "sum_net": net,
            "avg_net_pct": float(r["avg_net_pct"] or 0),
            "gross": gross,
            "near_zero": 1 if abs(gross) < 20.0 and net < 0 else 0,
            "gpn": 1 if gross > 0 and net <= 0 else 0,
        }
        accum[bk].append(row)

    out: Dict[BucketKey, BucketTradeStats] = {}
    for bk, lst in accum.items():
        if not lst:
            out[bk] = BucketTradeStats(bk, 0, 0.0, 0.0, 0.0, 0.0, 0, 0)
            continue
        n = len(lst)
        sn = sum(x["sum_net"] for x in lst)
        sg = sum(x["gross"] for x in lst)
        avg_pct = sum(x["avg_net_pct"] for x in lst) / n
        out[bk] = BucketTradeStats(
            bucket=bk,
            n_signals=n,
            sum_net_pnl=sn,
            avg_net_pnl=sn / n,
            avg_net_pnl_pct=avg_pct,
            sum_gross=sg,
            gross_pos_net_nonpos=sum(x["gpn"] for x in lst),
            near_zero_churn=sum(x["near_zero"] for x in lst),
        )
    return out


def fetch_pending_adaptive_actions(db: AnalyticsDB) -> List[Any]:
    db.migrate_adaptive_actions_columns()
    return db.fetchall(
        """
        SELECT action_id, action_ts, mode, scope, bucket_key, parameter_name,
               old_value, new_value, sample_size, confidence_score, action_status,
               reason_text, metrics_json, created_at, applied, reverted, evaluation_status
        FROM adaptive_actions
        WHERE IFNULL(evaluation_status, 'n/a') = 'pending'
          AND IFNULL(reverted, 0) = 0
          AND IFNULL(applied, 1) = 1
        ORDER BY action_ts ASC
        """
    )


def count_post_close_trades_in_bucket(db: AnalyticsDB, bucket: BucketKey, since_ts: str) -> int:
    vr = _vr_expr("s")
    sql = f"""
    SELECT COUNT(DISTINCT s.signal_id) AS n
    FROM signals s
    JOIN decision_logs c ON c.signal_id = s.signal_id AND c.decision_type = 'TRADE_CLOSE'
    WHERE s.side = 'BUY'
      AND c.decision_ts > ?
      AND (
        CASE
          WHEN CAST({vr} AS REAL) < 1 THEN 'lt1'
          WHEN CAST({vr} AS REAL) <= 2 THEN 'b12'
          ELSE 'gt2'
        END
      ) = ?
    """
    row = db.fetchone(sql, (since_ts, bucket))
    return int(row["n"] or 0) if row else 0


def avg_net_pct_in_bucket_after(db: AnalyticsDB, bucket: BucketKey, since_ts: str) -> Optional[float]:
    vr = _vr_expr("s")
    sql = f"""
    SELECT AVG(t.avg_net_pct) AS m
    FROM (
      SELECT s.signal_id, AVG(json_extract(c.details_json, '$.net_pnl_pct')) AS avg_net_pct
      FROM signals s
      JOIN decision_logs c ON c.signal_id = s.signal_id AND c.decision_type = 'TRADE_CLOSE'
      WHERE s.side = 'BUY'
        AND c.decision_ts > ?
        AND (
          CASE
            WHEN CAST({vr} AS REAL) < 1 THEN 'lt1'
            WHEN CAST({vr} AS REAL) <= 2 THEN 'b12'
            ELSE 'gt2'
          END
        ) = ?
      GROUP BY s.signal_id
    ) t
    """
    row = db.fetchone(sql, (since_ts, bucket))
    if not row or row["m"] is None:
        return None
    return float(row["m"])

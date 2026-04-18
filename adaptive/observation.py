from __future__ import annotations

import json
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from analytics.db import AnalyticsDB

from .buckets import BucketKey, volume_ratio_to_bucket


@dataclass
class BucketTradeStats:
    """Агрегаты по bucket из БД: signals + TRADE_CLOSE (реализованный net)."""

    bucket: BucketKey
    n_closed: int  # число BUY-сигналов с хотя бы одним CLOSE в окне
    sum_net_pnl: float
    avg_net_pnl: float
    avg_net_pnl_pct: float
    median_net_pnl_pct: float
    sum_gross: float
    gross_pos_net_nonpos: int
    near_zero_churn: int
    near_zero_rate: float
    prev_n_closed: int
    prev_avg_net_pnl_pct: float
    delta_avg_vs_prev: float
    confidence_sample: float  # 0..1 от объёма выборки
    outcome_15m_avg: Optional[float]  # справочно: signal_outcomes (не для revert-логики)


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


def _bucket_case_sql(vr: str = None) -> str:
    expr = vr or _vr_expr("s")
    return f"""CASE
          WHEN CAST({expr} AS REAL) < 1 THEN 'lt1'
          WHEN CAST({expr} AS REAL) <= 2 THEN 'b12'
          ELSE 'gt2'
        END"""


def _fetch_close_rows_for_window(
    db: AnalyticsDB, start_mod: str, end_mod: Optional[str]
) -> List[Any]:
    """
    Строки на уровне signal_id: реализованный PnL из decision_logs TRADE_CLOSE.
    Источник правды для adaptive: БД, не эвристики.
    """
    vr = _vr_expr("s")
    if end_mod is None:
        time_clause = "datetime(s.signal_ts) >= datetime('now', ?)"
        params: Tuple[Any, ...] = (start_mod,)
    else:
        time_clause = (
            "datetime(s.signal_ts) >= datetime('now', ?) "
            "AND datetime(s.signal_ts) < datetime('now', ?)"
        )
        params = (start_mod, end_mod)
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
      AND {time_clause}
    GROUP BY s.signal_id
    HAVING vr IS NOT NULL
    """
    return db.fetchall(sql, params)


def _rows_into_bucket_accum(rows: List[Any]) -> Dict[BucketKey, List[Dict[str, Any]]]:
    accum: Dict[BucketKey, List[Dict[str, Any]]] = {"lt1": [], "b12": [], "gt2": []}
    for r in rows:
        vr_val = float(r["vr"])
        bk = volume_ratio_to_bucket(vr_val)
        if bk is None:
            continue
        gross = float(r["sum_gross"] or 0)
        net = float(r["sum_net"] or 0)
        accum[bk].append(
            {
                "sum_net": net,
                "avg_net_pct": float(r["avg_net_pct"] or 0),
                "gross": gross,
                "near_zero": 1 if abs(gross) < 20.0 and net < 0 else 0,
                "gpn": 1 if gross > 0 and net <= 0 else 0,
            }
        )
    return accum


def _finalize_bucket(
    bk: BucketKey,
    lst: List[Dict[str, Any]],
    prev_lst: List[Dict[str, Any]],
    min_sample: int,
) -> BucketTradeStats:
    if not lst:
        prev_avg = (
            sum(x["avg_net_pct"] for x in prev_lst) / len(prev_lst) if prev_lst else 0.0
        )
        return BucketTradeStats(
            bucket=bk,
            n_closed=0,
            sum_net_pnl=0.0,
            avg_net_pnl=0.0,
            avg_net_pnl_pct=0.0,
            median_net_pnl_pct=0.0,
            sum_gross=0.0,
            gross_pos_net_nonpos=0,
            near_zero_churn=0,
            near_zero_rate=0.0,
            prev_n_closed=len(prev_lst),
            prev_avg_net_pnl_pct=prev_avg,
            delta_avg_vs_prev=0.0,
            confidence_sample=0.0,
            outcome_15m_avg=None,
        )
    n = len(lst)
    sn = sum(x["sum_net"] for x in lst)
    sg = sum(x["gross"] for x in lst)
    pcts = [x["avg_net_pct"] for x in lst]
    avg_pct = sum(pcts) / n
    med_pct = float(statistics.median(pcts)) if n else 0.0
    prev_n = len(prev_lst)
    prev_avg = sum(x["avg_net_pct"] for x in prev_lst) / prev_n if prev_n else 0.0
    conf = min(1.0, float(n) / float(max(1, min_sample)))
    return BucketTradeStats(
        bucket=bk,
        n_closed=n,
        sum_net_pnl=sn,
        avg_net_pnl=sn / n,
        avg_net_pnl_pct=avg_pct,
        median_net_pnl_pct=med_pct,
        sum_gross=sg,
        gross_pos_net_nonpos=sum(x["gpn"] for x in lst),
        near_zero_churn=sum(x["near_zero"] for x in lst),
        near_zero_rate=sum(x["near_zero"] for x in lst) / n,
        prev_n_closed=prev_n,
        prev_avg_net_pnl_pct=prev_avg,
        delta_avg_vs_prev=avg_pct - prev_avg,
        confidence_sample=conf,
        outcome_15m_avg=None,
    )


def fetch_bucket_outcome_15m_avgs(db: AnalyticsDB, observation_days: int) -> Dict[BucketKey, Optional[float]]:
    """Справочно: качество входов по forward outcome (БД), не используется в revert."""
    mod = f"-{int(observation_days)} days"
    vr = _vr_expr("s")
    bc = _bucket_case_sql(vr)
    sql = f"""
    SELECT {bc} AS bk, AVG(o.outcome_15m_pct) AS m, COUNT(*) AS n
    FROM signals s
    JOIN signal_outcomes o ON o.signal_id = s.signal_id
    WHERE s.side = 'BUY'
      AND datetime(s.signal_ts) >= datetime('now', ?)
      AND o.outcome_15m_pct IS NOT NULL
    GROUP BY bk
    """
    rows = db.fetchall(sql, (mod,))
    out: Dict[BucketKey, Optional[float]] = {"lt1": None, "b12": None, "gt2": None}
    for r in rows:
        bk = r["bk"]
        if bk in out and int(r["n"] or 0) > 0:
            out[bk] = float(r["m"])  # type: ignore[index]
    return out


def fetch_bucket_close_stats(
    db: AnalyticsDB, observation_days: int, min_sample_per_bucket: int
) -> Dict[BucketKey, BucketTradeStats]:
    """
    Текущее окно: последние D дней по signal_ts.
    Предыдущее окно: D дней непосредственно перед ним (тот же горизонт для сравнения в self-review).
    """
    d = int(observation_days)
    start_cur = f"-{d} days"
    end_cur = None
    start_prev = f"-{2 * d} days"
    end_prev = f"-{d} days"

    rows_cur = _fetch_close_rows_for_window(db, start_cur, end_cur)
    rows_prev = _fetch_close_rows_for_window(db, start_prev, end_prev)
    acc_cur = _rows_into_bucket_accum(rows_cur)
    acc_prev = _rows_into_bucket_accum(rows_prev)

    try:
        outcomes = fetch_bucket_outcome_15m_avgs(db, d)
    except Exception:
        outcomes = {"lt1": None, "b12": None, "gt2": None}

    out: Dict[BucketKey, BucketTradeStats] = {}
    for bk in ("lt1", "b12", "gt2"):
        st = _finalize_bucket(bk, acc_cur[bk], acc_prev[bk], int(min_sample_per_bucket))
        st.outcome_15m_avg = outcomes.get(bk)
        out[bk] = st
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

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from analytics.db import AnalyticsDB


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def insert_adaptive_action(
    db: AnalyticsDB,
    *,
    adaptive_mode: str,
    scope: str,
    bucket_key: str,
    parameter_name: str,
    old_value: Optional[float],
    new_value: Optional[float],
    sample_size: int,
    confidence_score: float,
    action_status: str,
    reason_text: str,
    metrics: Dict[str, Any],
    window_summary: Optional[Dict[str, Any]] = None,
    applied: bool = True,
    evaluation_status: str = "n/a",
) -> str:
    db.migrate_adaptive_actions_columns()
    action_id = str(uuid.uuid4())
    ts = _now_iso()
    db.execute(
        """
        INSERT INTO adaptive_actions (
          action_id, action_ts, mode, scope, bucket_key, parameter_name,
          old_value, new_value, confidence_score, sample_size, action_status,
          reason_text, metrics_json, window_summary_json, created_at,
          applied, reverted, evaluation_status
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?)
        """,
        (
            action_id,
            ts,
            adaptive_mode,
            scope,
            bucket_key,
            parameter_name,
            old_value,
            new_value,
            confidence_score,
            sample_size,
            action_status,
            reason_text,
            json.dumps(metrics, ensure_ascii=False),
            json.dumps(window_summary or {}, ensure_ascii=False),
            ts,
            1 if applied else 0,
            evaluation_status,
        ),
    )
    return action_id


def update_action_evaluation(
    db: AnalyticsDB,
    action_id: str,
    *,
    evaluation_status: str,
    metrics_extra: Optional[Dict[str, Any]] = None,
) -> None:
    db.migrate_adaptive_actions_columns()
    row = db.fetchone(
        "SELECT metrics_json FROM adaptive_actions WHERE action_id = ?", (action_id,)
    )
    merged: Dict[str, Any] = {}
    if row and row["metrics_json"]:
        try:
            merged = json.loads(row["metrics_json"])
        except Exception:
            merged = {}
    if metrics_extra:
        merged.update(metrics_extra)
    db.execute(
        "UPDATE adaptive_actions SET evaluation_status = ?, metrics_json = ? WHERE action_id = ?",
        (evaluation_status, json.dumps(merged, ensure_ascii=False), action_id),
    )


def mark_action_reverted(
    db: AnalyticsDB,
    action_id: str,
    reason: str,
    *,
    metrics_extra: Optional[Dict[str, Any]] = None,
) -> None:
    db.migrate_adaptive_actions_columns()
    merged = dict(metrics_extra or {})
    merged["revert_reason"] = reason
    update_action_evaluation(
        db,
        action_id,
        evaluation_status="worsened_reverted",
        metrics_extra=merged,
    )
    db.execute(
        "UPDATE adaptive_actions SET reverted = 1, action_status = 'reverted' WHERE action_id = ?",
        (action_id,),
    )

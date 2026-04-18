"""
Интеграция adaptive volume_ratio → ML threshold в торговом цикле (paper).

Использование в bot.py: см. adaptive/bot_hook.py и функцию prepare_ml_buy_decision.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from analytics.db import AnalyticsDB

from .config import AdaptiveConfig
from .integration import AdaptiveResolution, effective_ml_threshold_for_entry

# reason_code для decision_logs / signals
BLOCK_ADAPTIVE = "BLOCK_ADAPTIVE"
BLOCK_ADAPTIVE_BUCKET = "BLOCK_ADAPTIVE_BUCKET"


def _db(cfg: AdaptiveConfig) -> AnalyticsDB:
    return AnalyticsDB(cfg.db_path)


def fetch_adaptive_context_for_bucket(
    db: AnalyticsDB, bucket: Optional[str]
) -> Tuple[Optional[str], str]:
    """
    adaptive_action_id: активное pending-применение по bucket (если есть),
    иначе последнее applied действие по bucket (для аудита).
    adaptive_reason: reason_text из adaptive_actions.
    """
    if not bucket:
        return None, ""
    row = db.fetchone(
        """
        SELECT action_id, reason_text FROM adaptive_actions
        WHERE bucket_key = ? AND IFNULL(applied,0) = 1
          AND IFNULL(evaluation_status,'') = 'pending'
        ORDER BY action_ts DESC LIMIT 1
        """,
        (bucket,),
    )
    if row:
        return str(row["action_id"]), str(row["reason_text"] or "")
    row2 = db.fetchone(
        """
        SELECT action_id, reason_text FROM adaptive_actions
        WHERE bucket_key = ? AND IFNULL(applied,0) = 1
        ORDER BY action_ts DESC LIMIT 1
        """,
        (bucket,),
    )
    if row2:
        return str(row2["action_id"]), str(row2["reason_text"] or "")
    return None, ""


def build_adaptive_strategy_decision_details(
    *,
    db: AnalyticsDB,
    cfg: AdaptiveConfig,
    trading_mode: str,
    adaptive_mode: str,
    base_ml_threshold: float,
    volume_ratio: Optional[float],
    resolution: AdaptiveResolution,
) -> Dict[str, Any]:
    """Поля для decision_logs.details_json (BUY или adaptive BLOCK)."""
    bucket = resolution.bucket
    action_id, reason = fetch_adaptive_context_for_bucket(db, bucket)
    return {
        "adaptive_mode": adaptive_mode,
        "bucket": bucket,
        "base_ml_threshold": float(base_ml_threshold),
        "effective_ml_threshold": float(resolution.effective_ml_threshold),
        "bucket_blocked": bool(resolution.hard_block_bucket),
        "adaptive_action_id": action_id,
        "adaptive_reason": reason or resolution.meta.get("note") or "",
    }


def evaluate_ml_buy_with_adaptive(
    *,
    ml_prob: Optional[float],
    volume_ratio: Optional[float],
    trading_mode: str,
    adaptive_mode: str,
    base_ml_threshold: float,
    db: Optional[AnalyticsDB] = None,
    cfg: Optional[AdaptiveConfig] = None,
) -> Tuple[bool, Optional[str], AdaptiveResolution, Dict[str, Any]]:
    """
    Возвращает:
      allow_buy, reason_code_if_blocked, resolution, details_json_fragment
    reason_code: BLOCK_ADAPTIVE_BUCKET | BLOCK_ADAPTIVE | None
    """
    cfg = cfg or AdaptiveConfig.from_env()
    db = db or _db(cfg)
    res = effective_ml_threshold_for_entry(
        base_ml_threshold=base_ml_threshold,
        volume_ratio=volume_ratio,
        trading_mode=trading_mode,
        adaptive_mode=adaptive_mode,
        cfg=cfg,
    )
    details = build_adaptive_strategy_decision_details(
        db=db,
        cfg=cfg,
        trading_mode=trading_mode,
        adaptive_mode=adaptive_mode,
        base_ml_threshold=base_ml_threshold,
        volume_ratio=volume_ratio,
        resolution=res,
    )
    if trading_mode != "paper" or adaptive_mode != "paper":
        details["adaptive_note"] = "adaptive does not steer execution outside paper+paper"
    if res.hard_block_bucket:
        details["adaptive_block"] = "hard_bucket_block"
        return False, BLOCK_ADAPTIVE_BUCKET, res, details
    if ml_prob is None:
        details["adaptive_block"] = "ml_prob_missing"
        return False, BLOCK_ADAPTIVE, res, details
    if ml_prob < res.effective_ml_threshold:
        details["adaptive_block"] = "ml_below_effective_threshold"
        details["ml_prob"] = float(ml_prob)
        return False, BLOCK_ADAPTIVE, res, details
    details["ml_prob"] = float(ml_prob)
    return True, None, res, details

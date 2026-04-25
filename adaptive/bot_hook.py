"""
Куда встраивать adaptive в bot.py (paper, supertrend+ML).

В репозитории нет bot.py — используйте prepare_ml_buy_decision() из основного бота.

Место вставки: сразу после расчёта ml_prob и feature snapshot, ДО записи BUY / ордера.

Без этого вызова adaptive loop только наблюдает (journal + runtime JSON), но не управляет сделками.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from analytics.db import AnalyticsDB

from .config import AdaptiveConfig
from .trading_integration import evaluate_ml_buy_with_adaptive


def _extract_volume_ratio(feature_snapshot: Dict[str, Any]) -> Optional[float]:
    vr = feature_snapshot.get("volume_ratio")
    if vr is None:
        vr = feature_snapshot.get("vol_ratio")
    try:
        return float(vr) if vr is not None else None
    except (TypeError, ValueError):
        return None


def prepare_ml_buy_decision(
    *,
    ml_prob: Optional[float],
    feature_snapshot: Dict[str, Any],
    trading_mode: str,
    adaptive_mode: str,
    base_ml_threshold: float,
    db_path: Optional[str] = None,
) -> Dict[str, Any]:
    cfg = AdaptiveConfig.from_env()
    db = AnalyticsDB(Path(db_path)) if db_path else AnalyticsDB(cfg.db_path)
    vr_f = _extract_volume_ratio(feature_snapshot)
    allow, reason, res, details = evaluate_ml_buy_with_adaptive(
        ml_prob=ml_prob,
        volume_ratio=vr_f,
        trading_mode=trading_mode,
        adaptive_mode=adaptive_mode,
        base_ml_threshold=base_ml_threshold,
        db=db,
        cfg=cfg,
    )
    effective_ml_threshold = float(res.effective_ml_threshold)

    if trading_mode == "paper" and not res.hard_block_bucket:
        ticker = str(feature_snapshot.get("ticker") or "").upper()
        if ticker in ("BSPB", "CHMF") and vr_f is not None and vr_f < 1:
            if ml_prob is not None and float(ml_prob) >= 0.08:
                effective_ml_threshold = min(effective_ml_threshold, 0.08)
                allow = float(ml_prob) >= effective_ml_threshold
                reason = None if allow else reason
                details["adaptive_override"] = "ticker_lt1_soft_threshold"
                details["original_threshold"] = float(base_ml_threshold)
                details["effective_threshold"] = float(effective_ml_threshold)
                details["effective_ml_threshold"] = float(effective_ml_threshold)
                if allow:
                    details.pop("adaptive_block", None)

    return {
        "allow_buy": allow,
        "reason_code": reason,
        "details_adaptive": details,
        "effective_ml_threshold": effective_ml_threshold,
        "hard_block_bucket": res.hard_block_bucket,
    }

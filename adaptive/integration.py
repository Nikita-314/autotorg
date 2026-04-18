from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from .buckets import volume_ratio_to_bucket
from .config import AdaptiveConfig
from .state import AdaptiveStateStore


@dataclass
class AdaptiveResolution:
    """Результат для точки входа ML: итоговый порог и жёсткий блок bucket (paper)."""

    base_ml_threshold: float
    effective_ml_threshold: float
    hard_block_bucket: bool
    bucket: Optional[str]
    meta: Dict[str, Any]


def effective_ml_threshold_for_entry(
    *,
    base_ml_threshold: float,
    volume_ratio: Optional[float],
    trading_mode: str,
    adaptive_mode: str,
    store: Optional[AdaptiveStateStore] = None,
    cfg: Optional[AdaptiveConfig] = None,
) -> AdaptiveResolution:
    """
    Вызывать из бота перед сравнением ml_prob с порогом.
    Только ADAPTIVE_MODE=paper + TRADING_MODE=paper применяют модификаторы из state.
    """
    cfg = cfg or AdaptiveConfig.from_env()
    meta: Dict[str, Any] = {
        "adaptive_mode": adaptive_mode,
        "trading_mode": trading_mode,
    }
    bucket = volume_ratio_to_bucket(volume_ratio)
    meta["volume_ratio"] = volume_ratio
    meta["bucket"] = bucket

    if bucket is None:
        return AdaptiveResolution(
            base_ml_threshold=base_ml_threshold,
            effective_ml_threshold=base_ml_threshold,
            hard_block_bucket=False,
            bucket=None,
            meta=meta,
        )

    store = store or AdaptiveStateStore(cfg.runtime_state_path)
    state = store.load(base_ml_threshold)

    if adaptive_mode == "off":
        meta["note"] = "ADAPTIVE_MODE=off"
        return AdaptiveResolution(
            base_ml_threshold=base_ml_threshold,
            effective_ml_threshold=float(base_ml_threshold),
            hard_block_bucket=False,
            bucket=bucket,
            meta=meta,
        )
    if adaptive_mode != "paper" or trading_mode != "paper":
        meta["note"] = "adaptive execution requires ADAPTIVE_MODE=paper and TRADING_MODE=paper"
        return AdaptiveResolution(
            base_ml_threshold=base_ml_threshold,
            effective_ml_threshold=float(base_ml_threshold),
            hard_block_bucket=False,
            bucket=bucket,
            meta=meta,
        )

    blocked = bool(state.blocked.get(bucket, False))
    if blocked:
        meta["hard_block_bucket"] = True
        return AdaptiveResolution(
            base_ml_threshold=base_ml_threshold,
            effective_ml_threshold=1.0,
            hard_block_bucket=True,
            bucket=bucket,
            meta=meta,
        )

    eff = float(state.effective_threshold.get(bucket, base_ml_threshold))
    meta["effective_ml_threshold"] = eff
    return AdaptiveResolution(
        base_ml_threshold=base_ml_threshold,
        effective_ml_threshold=eff,
        hard_block_bucket=False,
        bucket=bucket,
        meta=meta,
    )

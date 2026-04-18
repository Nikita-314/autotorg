"""
Paper-only фильтр входа при аномально высоком volume_ratio.

Включается отдельным флагом в окружении (см. load_paper_high_volume_settings).
Аналитика: reason_code BLOCK_HIGH_VOLUME_RATIO + volume_ratio в details.

Интеграция: вызывать из места, где уже решён long-кандидат и посчитан ml_prob
(после Supertrend / до оформления BUY), только при TRADING_MODE=paper.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

LOGGER = logging.getLogger("paper-entry-volume")

BLOCK_HIGH_VOLUME_RATIO_REASON = "BLOCK_HIGH_VOLUME_RATIO"


def _truthy(val: Optional[str]) -> bool:
    if val is None:
        return False
    return val.strip().lower() in ("1", "true", "yes", "on")


@dataclass(frozen=True)
class PaperHighVolumeSettings:
    """Настройки из env (только смысл для paper — проверка режима снаружи)."""

    enabled: bool
    ratio_limit: float
    action: str  # "block" | "raise_ml_threshold"
    ml_rel_bump: float  # например 0.25 => порог ML * 1.25

    @staticmethod
    def from_env() -> "PaperHighVolumeSettings":
        action = (os.getenv("PAPER_HIGH_VOLUME_ACTION") or "block").strip().lower()
        if action not in ("block", "raise_ml_threshold"):
            action = "block"
        return PaperHighVolumeSettings(
            enabled=_truthy(os.getenv("PAPER_ENABLE_HIGH_VOLUME_ENTRY_FILTER")),
            ratio_limit=float(os.getenv("PAPER_HIGH_VOLUME_RATIO_LIMIT", "2.0")),
            action=action,
            ml_rel_bump=float(os.getenv("PAPER_HIGH_VOLUME_ML_THRESHOLD_REL_BUMP", "0.25")),
        )


@dataclass(frozen=True)
class PaperHighVolumeGateResult:
    """Результат проверки: блокируем ли BUY и какой порог ML использовать."""

    triggered: bool
    block_buy: bool
    reason_code: Optional[str]
    volume_ratio: Optional[float]
    base_ml_threshold: float
    effective_ml_threshold: float
    details: Dict[str, Any]


def extract_volume_ratio(snapshot: Optional[Dict[str, Any]]) -> Optional[float]:
    """
    Достаёт отношение объёма из снапшота признаков.
    В фичах проекта поле называется vol_ratio; в analytics snapshot — volume_ratio.
    """
    if not snapshot:
        return None
    for key in ("volume_ratio", "vol_ratio"):
        if key in snapshot and snapshot[key] is not None:
            try:
                return float(snapshot[key])
            except (TypeError, ValueError):
                return None
    return None


def build_block_details(
    *,
    volume_ratio: float,
    action: str,
    base_ml_threshold: float,
    effective_ml_threshold: float,
    ml_prob: Optional[float],
) -> Dict[str, Any]:
    """Полезная нагрузка для decision_logs / signal snapshot."""
    return {
        "volume_ratio": volume_ratio,
        "filter": "paper_high_volume",
        "action": action,
        "base_ml_threshold": base_ml_threshold,
        "effective_ml_threshold": effective_ml_threshold,
        "ml_prob": ml_prob,
        "reason_code": BLOCK_HIGH_VOLUME_RATIO_REASON,
    }


def apply_paper_high_volume_entry_gate(
    *,
    trading_mode: str,
    settings: Optional[PaperHighVolumeSettings] = None,
    volume_ratio: Optional[float],
    ml_prob: Optional[float],
    base_ml_threshold: float,
) -> PaperHighVolumeGateResult:
    """
    Paper-only: при включённом флаге и volume_ratio > ratio_limit
    - block: всегда block_buy=True (кандидат в long не должен стать BUY)
    - raise_ml_threshold: block_buy, если ml_prob < effective порога
    """
    cfg = settings or PaperHighVolumeSettings.from_env()
    base_t = float(base_ml_threshold)

    def _ok(vr: Optional[float]) -> PaperHighVolumeGateResult:
        eff = base_t
        return PaperHighVolumeGateResult(
            triggered=False,
            block_buy=False,
            reason_code=None,
            volume_ratio=vr,
            base_ml_threshold=base_t,
            effective_ml_threshold=eff,
            details={},
        )

    if trading_mode.strip().lower() != "paper" or not cfg.enabled:
        return _ok(volume_ratio)

    if volume_ratio is None or volume_ratio <= cfg.ratio_limit:
        return _ok(volume_ratio)

    eff = min(0.99, base_t * (1.0 + max(0.0, cfg.ml_rel_bump)))

    if cfg.action == "raise_ml_threshold":
        if ml_prob is None:
            block = True
        else:
            block = ml_prob < eff
        if block:
            details = build_block_details(
                volume_ratio=volume_ratio,
                action=cfg.action,
                base_ml_threshold=base_t,
                effective_ml_threshold=eff,
                ml_prob=ml_prob,
            )
            LOGGER.info(
                "%s ml_prob=%s base=%.4f eff=%.4f vr=%.4f action=%s",
                BLOCK_HIGH_VOLUME_RATIO_REASON,
                ml_prob,
                base_t,
                eff,
                volume_ratio,
                cfg.action,
            )
            return PaperHighVolumeGateResult(
                triggered=True,
                block_buy=True,
                reason_code=BLOCK_HIGH_VOLUME_RATIO_REASON,
                volume_ratio=volume_ratio,
                base_ml_threshold=base_t,
                effective_ml_threshold=eff,
                details=details,
            )
        return PaperHighVolumeGateResult(
            triggered=True,
            block_buy=False,
            reason_code=None,
            volume_ratio=volume_ratio,
            base_ml_threshold=base_t,
            effective_ml_threshold=eff,
            details={
                "volume_ratio": volume_ratio,
                "filter": "paper_high_volume",
                "action": cfg.action,
                "note": "volume spike but ml_prob clears raised threshold",
                "ml_prob": ml_prob,
                "base_ml_threshold": base_t,
                "effective_ml_threshold": eff,
            },
        )

    # cfg.action == "block"
    details = build_block_details(
        volume_ratio=volume_ratio,
        action="block",
        base_ml_threshold=base_t,
        effective_ml_threshold=base_t,
        ml_prob=ml_prob,
    )
    LOGGER.info(
        "%s vr=%.4f limit=%.4f action=block ml_prob=%s",
        BLOCK_HIGH_VOLUME_RATIO_REASON,
        volume_ratio,
        cfg.ratio_limit,
        ml_prob,
    )
    return PaperHighVolumeGateResult(
        triggered=True,
        block_buy=True,
        reason_code=BLOCK_HIGH_VOLUME_RATIO_REASON,
        volume_ratio=volume_ratio,
        base_ml_threshold=base_t,
        effective_ml_threshold=base_t,
        details=details,
    )

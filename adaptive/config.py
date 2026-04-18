from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _truthy(val: str | None) -> bool:
    if val is None:
        return False
    return val.strip().lower() in ("1", "true", "yes", "on")


@dataclass(frozen=True)
class AdaptiveConfig:
    adaptive_mode: str  # off | shadow | paper
    trading_mode: str
    db_path: Path
    balance_state_path: Path
    runtime_state_path: Path
    observation_days: int
    min_sample_per_bucket: int
    eval_min_post_trades: int
    bad_avg_net_pct: float
    good_avg_net_pct: float
    very_bad_avg_net_pct: float
    block_min_sample: int
    threshold_step: float
    max_rel_change: float  # ±15% default -> 0.15
    loop_interval_sec: int
    telegram_chat_id: str | None
    bucket_cooldown_hours: int
    # Пока есть любое applied+pending — не начинать новое paper-применение (кроме evaluate)
    stall_new_until_any_pending_eval: bool

    @staticmethod
    def from_env() -> "AdaptiveConfig":
        root = Path(__file__).resolve().parent.parent
        return AdaptiveConfig(
            adaptive_mode=(os.getenv("ADAPTIVE_MODE") or "off").strip().lower(),
            trading_mode=(os.getenv("TRADING_MODE") or "").strip().lower(),
            db_path=Path(os.getenv("ADAPTIVE_DB_PATH") or root / "analytics.db"),
            balance_state_path=Path(
                os.getenv("PAPER_BALANCE_STATE_PATH") or root / "paper_balance_state.json"
            ),
            runtime_state_path=Path(
                os.getenv("ADAPTIVE_RUNTIME_STATE_PATH") or root / "adaptive_runtime_state.json"
            ),
            observation_days=int(os.getenv("ADAPTIVE_OBSERVATION_DAYS", "14")),
            min_sample_per_bucket=int(os.getenv("ADAPTIVE_MIN_SAMPLE_PER_BUCKET", "10")),
            eval_min_post_trades=int(os.getenv("ADAPTIVE_EVAL_MIN_POST_TRADES", "8")),
            bad_avg_net_pct=float(os.getenv("ADAPTIVE_BAD_AVG_NET_PCT", "-0.25")),
            good_avg_net_pct=float(os.getenv("ADAPTIVE_GOOD_AVG_NET_PCT", "0.08")),
            very_bad_avg_net_pct=float(os.getenv("ADAPTIVE_VERY_BAD_AVG_NET_PCT", "-0.45")),
            block_min_sample=int(os.getenv("ADAPTIVE_BLOCK_MIN_SAMPLE", "18")),
            threshold_step=float(os.getenv("ADAPTIVE_THRESHOLD_STEP", "0.015")),
            max_rel_change=float(os.getenv("ADAPTIVE_MAX_REL_THRESHOLD_CHANGE", "0.15")),
            loop_interval_sec=int(os.getenv("ADAPTIVE_LOOP_INTERVAL_SEC", "600")),
            telegram_chat_id=os.getenv("TELEGRAM_CHAT_ID"),
            bucket_cooldown_hours=int(os.getenv("ADAPTIVE_BUCKET_COOLDOWN_HOURS", "12")),
            stall_new_until_any_pending_eval=_truthy(
                os.getenv("ADAPTIVE_STALL_NEW_UNTIL_PENDING_EVAL", "true")
            ),
        )

    def allows_shadow_work(self) -> bool:
        return self.adaptive_mode in ("shadow", "paper")

    def allows_apply(self) -> bool:
        return self.adaptive_mode == "paper" and self.trading_mode == "paper"

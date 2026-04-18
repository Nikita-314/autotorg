from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

from .buckets import BucketKey, BUCKET_LABELS
from .config import AdaptiveConfig
from .observation import BucketTradeStats


RecommendationType = Literal["none", "raise_threshold", "lower_threshold", "block_bucket", "unblock_bucket"]


@dataclass
class BucketRecommendation:
    bucket: BucketKey
    rec_type: RecommendationType
    confidence: float
    reason: str


def diagnose_bucket(
    cfg: AdaptiveConfig,
    stats: BucketTradeStats,
    currently_blocked: bool,
) -> BucketRecommendation:
    label = BUCKET_LABELS[stats.bucket]
    n = stats.n_signals
    if n < cfg.min_sample_per_bucket:
        return BucketRecommendation(
            bucket=stats.bucket,
            rec_type="none",
            confidence=0.0,
            reason=f"bucket {label}: sample {n} < min {cfg.min_sample_per_bucket}, no change",
        )

    avg_pct = stats.avg_net_pnl_pct

    if avg_pct <= cfg.very_bad_avg_net_pct and n >= cfg.block_min_sample and not currently_blocked:
        return BucketRecommendation(
            bucket=stats.bucket,
            rec_type="block_bucket",
            confidence=min(1.0, n / (cfg.block_min_sample * 2)),
            reason=(
                f"bucket {label}: very bad avg_net_pnl_pct={avg_pct:.3f}% over {n} signals "
                f"(threshold {cfg.very_bad_avg_net_pct}%), recommend block in paper"
            ),
        )

    if avg_pct < cfg.bad_avg_net_pct:
        return BucketRecommendation(
            bucket=stats.bucket,
            rec_type="raise_threshold",
            confidence=min(1.0, (-avg_pct) * n / 50.0),
            reason=(
                f"bucket {label}: weak avg_net_pnl_pct={avg_pct:.3f}% over {n} signals "
                f"(bad<{cfg.bad_avg_net_pct}%), raise ML threshold slightly"
            ),
        )

    if avg_pct > cfg.good_avg_net_pct and n >= cfg.min_sample_per_bucket:
        return BucketRecommendation(
            bucket=stats.bucket,
            rec_type="lower_threshold",
            confidence=min(1.0, avg_pct * n / 20.0),
            reason=(
                f"bucket {label}: good avg_net_pnl_pct={avg_pct:.3f}% over {n} signals, "
                f"lower ML threshold slightly within safe bounds"
            ),
        )

    if currently_blocked and avg_pct > cfg.bad_avg_net_pct * 0.5 and n >= cfg.min_sample_per_bucket:
        return BucketRecommendation(
            bucket=stats.bucket,
            rec_type="unblock_bucket",
            confidence=0.55,
            reason=f"bucket {label}: blocked but metrics improved (avg {avg_pct:.3f}%), suggest unblock",
        )

    return BucketRecommendation(
        bucket=stats.bucket,
        rec_type="none",
        confidence=0.2,
        reason=f"bucket {label}: neutral avg_net_pnl_pct={avg_pct:.3f}% (n={n}), hold",
    )

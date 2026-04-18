from __future__ import annotations

from typing import Dict, List

from .buckets import BUCKET_LABELS, BucketKey
from .config import AdaptiveConfig
from .observation import BucketTradeStats, PaperBalanceSnapshot


def build_self_review_report(
    cfg: AdaptiveConfig,
    bucket_stats: Dict[BucketKey, BucketTradeStats],
    balance: PaperBalanceSnapshot | None,
) -> str:
    """
    Explainable «работа над ошибками»: KPI по bucket, churn, gross+ / net-.
    """
    lines: List[str] = []
    lines.append("=== Adaptive self-review (explainable) ===")
    lines.append(f"Window: last {cfg.observation_days}d BUY→CLOSE joins")
    if balance:
        lines.append(
            f"Paper KPI: realized_net_pnl_rub={balance.realized_net_pnl_rub:.2f} "
            f"equity_rub={balance.equity_rub:.2f} commission={balance.realized_commission_rub:.2f}"
        )
        if balance.last_reset_at:
            lines.append(f"last_reset_at={balance.last_reset_at}")

    for bk in ("lt1", "b12", "gt2"):
        st = bucket_stats[bk]
        label = BUCKET_LABELS[bk]
        lines.append(
            f"- volume_ratio {label}: n={st.n_signals} avg_net%={st.avg_net_pnl_pct:.3f}% "
            f"avg_net_rub={st.avg_net_pnl:.2f} gross+&net<=0: {st.gross_pos_net_nonpos} "
            f"near_zero_churn: {st.near_zero_churn}"
        )
        if st.n_signals >= cfg.min_sample_per_bucket:
            if st.avg_net_pnl_pct < cfg.bad_avg_net_pct:
                lines.append(
                    f"  → weak bucket by net% (threshold bad={cfg.bad_avg_net_pct}): "
                    f"consider raising ML threshold or block if very bad."
                )
            elif st.avg_net_pnl_pct > cfg.good_avg_net_pct:
                lines.append(
                    f"  → strong bucket (good>{cfg.good_avg_net_pct}): may slightly lower threshold."
                )
        else:
            lines.append("  → insufficient sample for adaptation")

    lines.append("=== End self-review ===")
    return "\n".join(lines)

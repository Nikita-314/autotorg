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
    DB-first self-review: метрики из TRADE_CLOSE по bucket + сравнение с предыдущим окном того же размера;
    outcome_15m — справочно из signal_outcomes (не двигает порог).
    """
    lines: List[str] = []
    lines.append("=== Adaptive self-review (DB-first) ===")
    lines.append(
        f"Primary: signals(BUY) JOIN decision_logs(TRADE_CLOSE) — net_pnl / net_pnl_pct из details_json; "
        f"window={cfg.observation_days}d vs prev {cfg.observation_days}d"
    )
    lines.append(
        "Secondary (reference): signal_outcomes.outcome_15m_pct — forward proxy, not used in policy/revert"
    )
    if balance:
        lines.append(
            f"Paper balance file (KPI cross-check): net={balance.realized_net_pnl_rub:.2f} "
            f"equity={balance.equity_rub:.2f} comm={balance.realized_commission_rub:.2f}"
        )
        if balance.last_reset_at:
            lines.append(f"last_reset_at={balance.last_reset_at}")

    for bk in ("lt1", "b12", "gt2"):
        st = bucket_stats[bk]
        label = BUCKET_LABELS[bk]
        o15 = st.outcome_15m_avg
        o15s = f"{o15:.3f}%" if o15 is not None else "n/a"
        lines.append(
            f"- bucket {label}: closed_trades={st.n_closed} "
            f"avg_net%={st.avg_net_pnl_pct:.3f}% median_net%={st.median_net_pnl_pct:.3f}% "
            f"near_zero_rate={st.near_zero_rate:.2%} (n={st.near_zero_churn}) "
            f"gross+&net<=0={st.gross_pos_net_nonpos} "
            f"confidence~{st.confidence_sample:.2f} (n vs min_sample={cfg.min_sample_per_bucket})"
        )
        lines.append(
            f"    prev_window: n={st.prev_n_closed} avg_net%={st.prev_avg_net_pnl_pct:.3f}% "
            f"Δavg_vs_prev={st.delta_avg_vs_prev:+.3f}%"
        )
        lines.append(f"    ref outcome_15m_avg={o15s}")
        if st.n_closed < cfg.min_sample_per_bucket:
            lines.append(
                f"  → DB sparse for policy: n<{cfg.min_sample_per_bucket} (no threshold move from this bucket)"
            )
        elif st.avg_net_pnl_pct < cfg.bad_avg_net_pct:
            lines.append(
                f"  → weak realized net% vs threshold {cfg.bad_avg_net_pct}% (DB-driven diagnosis path)"
            )
        elif st.avg_net_pnl_pct > cfg.good_avg_net_pct:
            lines.append(f"  → strong realized net% vs threshold {cfg.good_avg_net_pct}%")

    lines.append("=== End self-review ===")
    return "\n".join(lines)

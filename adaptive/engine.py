from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from analytics.db import AnalyticsDB

from .buckets import BUCKET_LABELS, BucketKey
from .config import AdaptiveConfig
from .diagnosis import BucketRecommendation, diagnose_bucket
from .journal import insert_adaptive_action, mark_action_reverted, update_action_evaluation
from .observation import (
    avg_net_pct_in_bucket_after,
    count_post_close_trades_in_bucket,
    fetch_bucket_close_stats,
    fetch_pending_adaptive_actions,
    load_paper_balance,
)
from .self_review import build_self_review_report
from .state import AdaptiveRuntimeState, AdaptiveStateStore

LOGGER = logging.getLogger("adaptive-engine")


class AdaptiveEngine:
    """Один безопасный цикл: observe → diagnose → (shadow/paper) → evaluate pending."""

    def __init__(self, cfg: Optional[AdaptiveConfig] = None):
        self.cfg = cfg or AdaptiveConfig.from_env()
        self.db = AnalyticsDB(self.cfg.db_path)
        self.store = AdaptiveStateStore(self.cfg.runtime_state_path)

    def step(self) -> Dict[str, Any]:
        if self.cfg.adaptive_mode == "off":
            return {"status": "skipped", "reason": "ADAPTIVE_MODE=off"}

        self.db.migrate_adaptive_actions_columns()
        base_ml = float(os.getenv("ML_PROB_THRESHOLD", "0.63"))
        state = self.store.load(base_ml)
        balance = load_paper_balance(self.cfg.balance_state_path)
        buckets = fetch_bucket_close_stats(self.db, self.cfg.observation_days)
        report = build_self_review_report(self.cfg, buckets, balance)

        evaluated = self._evaluate_pending(state)
        applied = self._maybe_apply_one(state, buckets, base_ml)

        self.store.save(state)
        out: Dict[str, Any] = {
            "status": "ok",
            "adaptive_mode": self.cfg.adaptive_mode,
            "evaluated_actions": evaluated,
            "applied_recommendation": applied,
            "self_review_excerpt": report[:2000],
        }
        if applied and applied.get("paper_applied"):
            out["telegram_hint"] = (
                f"Adaptive change applied: bucket={applied.get('bucket')} "
                f"{applied.get('parameter')} {applied.get('old')} → {applied.get('new')}"
            )
        return out

    def _bucket_has_pending(self, bucket: str) -> bool:
        row = self.db.fetchone(
            """
            SELECT 1 AS x FROM adaptive_actions
            WHERE IFNULL(evaluation_status,'') = 'pending'
              AND bucket_key = ?
            LIMIT 1
            """,
            (bucket,),
        )
        return row is not None

    def _bucket_in_cooldown(self, bucket: str) -> bool:
        row = self.db.fetchone(
            "SELECT action_ts FROM adaptive_actions WHERE bucket_key = ? ORDER BY action_ts DESC LIMIT 1",
            (bucket,),
        )
        if not row or not row["action_ts"]:
            return False
        try:
            ts = datetime.fromisoformat(str(row["action_ts"]).replace("Z", "+00:00"))
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            return ts > datetime.now(timezone.utc) - timedelta(hours=self.cfg.bucket_cooldown_hours)
        except Exception:
            return False

    def _evaluate_pending(self, state: AdaptiveRuntimeState) -> List[str]:
        """
        Baseline: metrics.pre_avg_net_pnl_pct at apply time (bucket avg net% in observation window).
        Post: same metric recomputed only on closes strictly after action_ts in that bucket.
        improved: post > pre + 0.02; worsened: post < pre - 0.02 → revert runtime state;
        inconclusive: else or missing data.
        """
        out: List[str] = []
        pending = fetch_pending_adaptive_actions(self.db)
        for row in pending:
            action_id = row["action_id"]
            since_ts = row["action_ts"]
            bk = row["bucket_key"]
            if not bk:
                continue
            n_post = count_post_close_trades_in_bucket(self.db, bk, since_ts)  # type: ignore[arg-type]
            if n_post < self.cfg.eval_min_post_trades:
                continue
            post = avg_net_pct_in_bucket_after(self.db, bk, since_ts)  # type: ignore[arg-type]
            try:
                metrics = json.loads(row["metrics_json"] or "{}")
            except Exception:
                metrics = {}
            pre = metrics.get("pre_avg_net_pnl_pct")
            if pre is None or post is None:
                update_action_evaluation(self.db, action_id, evaluation_status="inconclusive")
                out.append(f"{action_id}: inconclusive (missing pre/post)")
                continue
            eps = 0.02
            if post < pre - eps:
                self._revert_action(state, row)
                mark_action_reverted(
                    self.db,
                    action_id,
                    f"post_avg {post:.4f} < pre_avg {pre:.4f}",
                    metrics_extra={"post_avg_net_pnl_pct": post, "pre_avg_net_pnl_pct": pre},
                )
                out.append(f"{action_id}: worsened → reverted")
            elif post > pre + eps:
                update_action_evaluation(
                    self.db,
                    action_id,
                    evaluation_status="improved",
                    metrics_extra={"post_avg_net_pnl_pct": post, "pre_avg_net_pnl_pct": pre},
                )
                out.append(f"{action_id}: improved → kept")
            else:
                update_action_evaluation(
                    self.db,
                    action_id,
                    evaluation_status="inconclusive",
                    metrics_extra={"post_avg_net_pnl_pct": post, "pre_avg_net_pnl_pct": pre},
                )
                out.append(f"{action_id}: inconclusive")
        return out

    def _revert_action(self, state: AdaptiveRuntimeState, row: Any) -> None:
        param = row["parameter_name"]
        bk = row["bucket_key"]
        old_v = row["old_value"]
        if param == "bucket_blocked":
            state.blocked[bk] = False
        elif param == "effective_ml_prob_threshold" and bk in state.effective_threshold:
            state.effective_threshold[bk] = float(old_v) if old_v is not None else state.base_ml_threshold

    def _any_pending_applied_globally(self) -> bool:
        """Есть незавершённая оценка applied-действия (любой bucket)."""
        row = self.db.fetchone(
            """
            SELECT 1 AS x FROM adaptive_actions
            WHERE IFNULL(evaluation_status,'') = 'pending'
              AND IFNULL(applied,0) = 1
            LIMIT 1
            """
        )
        return row is not None

    @staticmethod
    def _threshold_within_safe_band(base: float, value: float, max_rel: float) -> bool:
        lo = base * (1.0 - max_rel)
        hi = base * (1.0 + max_rel)
        return lo - 1e-9 <= value <= hi + 1e-9

    def _maybe_apply_one(
        self,
        state: AdaptiveRuntimeState,
        buckets: Dict[BucketKey, Any],
        base_ml: float,
    ) -> Optional[Dict[str, Any]]:
        if not self.cfg.allows_shadow_work():
            return None

        scored: List[tuple[float, BucketKey, BucketRecommendation]] = []
        for bk in ("gt2", "b12", "lt1"):
            st = buckets[bk]
            rec = diagnose_bucket(self.cfg, st, bool(state.blocked.get(bk)))
            if rec.rec_type != "none":
                scored.append((st.avg_net_pnl_pct, bk, rec))
        scored.sort(key=lambda x: x[0])

        for _, bk, rec in scored:
            if self._bucket_has_pending(bk):
                continue
            if self._bucket_in_cooldown(bk):
                continue
            if rec.rec_type == "none":
                continue
            return self._apply_recommendation(state, bk, rec, buckets[bk], base_ml)
        return None

    def _apply_recommendation(
        self,
        state: AdaptiveRuntimeState,
        bk: BucketKey,
        rec: BucketRecommendation,
        st: Any,
        base_ml: float,
    ) -> Dict[str, Any]:
        old_eff = float(state.effective_threshold.get(bk, base_ml))
        metrics = {
            "kpi": "realized_net_balance_proxy",
            "baseline_metric": "avg_net_pnl_pct_in_bucket_before_change",
            "baseline_value": st.avg_net_pnl_pct,
            "baseline_window_days": self.cfg.observation_days,
            "baseline_description": (
                f"BUY→TRADE_CLOSE aggregates in bucket {BUCKET_LABELS[bk]} over last "
                f"{self.cfg.observation_days}d (pre-change diagnostic window)"
            ),
            "pre_avg_net_pnl_pct": st.avg_net_pnl_pct,
            "pre_n_signals": st.n_signals,
            "pre_sum_net_pnl": st.sum_net_pnl,
            "gross_pos_net_nonpos": st.gross_pos_net_nonpos,
            "near_zero_churn": st.near_zero_churn,
            "post_window_rule": (
                "avg(json_extract net_pnl_pct) for BUY signals in same bucket with "
                "TRADE_CLOSE.decision_ts > action_ts; min closes="
                f"{self.cfg.eval_min_post_trades}"
            ),
        }

        applied_requested = self.cfg.allows_apply()
        applied = applied_requested
        if (
            self.cfg.stall_new_until_any_pending_eval
            and applied_requested
            and self._any_pending_applied_globally()
        ):
            applied = False
            LOGGER.info(
                "Adaptive paper apply downgraded to shadow: unresolved pending evaluation exists"
            )
        mode = self.cfg.adaptive_mode
        param = "effective_ml_prob_threshold"
        insert_old: Optional[float] = old_eff
        insert_new: Optional[float] = old_eff

        if rec.rec_type == "raise_threshold":
            insert_new = AdaptiveRuntimeState.clamp_threshold(
                base_ml, old_eff + self.cfg.threshold_step, self.cfg.max_rel_change
            )
            if applied:
                state.effective_threshold[bk] = float(insert_new)
        elif rec.rec_type == "lower_threshold":
            insert_new = AdaptiveRuntimeState.clamp_threshold(
                base_ml, old_eff - self.cfg.threshold_step, self.cfg.max_rel_change
            )
            if applied:
                state.effective_threshold[bk] = float(insert_new)
        elif rec.rec_type == "block_bucket":
            param = "bucket_blocked"
            insert_old, insert_new = 0.0, 1.0
            if applied:
                state.blocked[bk] = True
        elif rec.rec_type == "unblock_bucket":
            param = "bucket_blocked"
            insert_old, insert_new = 1.0, 0.0
            if applied:
                state.blocked[bk] = False
        else:
            return {"bucket": bk, "skipped": True}

        if param == "effective_ml_prob_threshold" and not self._threshold_within_safe_band(
            base_ml, float(insert_new), self.cfg.max_rel_change
        ):
            LOGGER.warning("Adaptive threshold out of safe band, skip: %s", insert_new)
            return {"bucket": bk, "skipped": True, "reason": "threshold_out_of_safe_band"}

        if not applied:
            insert_adaptive_action(
                self.db,
                adaptive_mode=mode,
                scope="volume_ratio_bucket",
                bucket_key=bk,
                parameter_name=param,
                old_value=insert_old,
                new_value=insert_new,
                sample_size=st.n_signals,
                confidence_score=rec.confidence,
                action_status="shadow_recommendation",
                reason_text=rec.reason,
                metrics=metrics,
                applied=False,
                evaluation_status="n/a",
            )
            LOGGER.info("Adaptive shadow recommendation: %s", rec.reason)
            return {"bucket": bk, "shadow": True, "reason": rec.reason}

        if applied and self.cfg.trading_mode != "paper":
            LOGGER.error("Refusing adaptive apply: TRADING_MODE is not paper")
            return {"bucket": bk, "skipped": True, "reason": "not_paper_trading_mode"}

        action_id = insert_adaptive_action(
            self.db,
            adaptive_mode=mode,
            scope="volume_ratio_bucket",
            bucket_key=bk,
            parameter_name=param,
            old_value=insert_old,
            new_value=insert_new,
            sample_size=st.n_signals,
            confidence_score=rec.confidence,
            action_status="applied",
            reason_text=rec.reason,
            metrics=metrics,
            applied=True,
            evaluation_status="pending",
        )
        LOGGER.info("Adaptive applied (paper): %s | %s -> %s id=%s", rec.reason, insert_old, insert_new, action_id)
        return {
            "bucket": bk,
            "paper_applied": True,
            "parameter": param,
            "old": insert_old,
            "new": insert_new,
            "adaptive_action_id": action_id,
        }

    def adaptive_status_text(self) -> str:
        base_ml = float(os.getenv("ML_PROB_THRESHOLD", "0.63"))
        state = self.store.load(base_ml)
        bal = load_paper_balance(self.cfg.balance_state_path)
        lines = [
            "=== Adaptive status ===",
            f"ADAPTIVE_MODE={self.cfg.adaptive_mode}",
            f"TRADING_MODE={self.cfg.trading_mode}",
            f"ML_PROB_THRESHOLD (base)={state.base_ml_threshold}",
            f"effective_ml by bucket: lt1={state.effective_threshold['lt1']:.4f} | "
            f"b12={state.effective_threshold['b12']:.4f} | gt2={state.effective_threshold['gt2']:.4f}",
            f"bucket_blocked: lt1={state.blocked['lt1']} b12={state.blocked['b12']} gt2={state.blocked['gt2']}",
            f"stall_new_until_pending_eval={self.cfg.stall_new_until_any_pending_eval}",
        ]
        if bal:
            lines.append(
                f"paper KPI: net={bal.realized_net_pnl_rub:.2f} equity={bal.equity_rub:.2f} "
                f"gross={bal.realized_gross_pnl_rub:.2f} comm={bal.realized_commission_rub:.2f}"
            )
            if bal.last_reset_at:
                lines.append(f"last_reset_at={bal.last_reset_at}")

        pend = self.db.fetchall(
            """
            SELECT action_id, action_ts, bucket_key, parameter_name, old_value, new_value, reason_text
            FROM adaptive_actions
            WHERE IFNULL(evaluation_status,'') = 'pending' AND IFNULL(applied,0) = 1
            ORDER BY action_ts ASC
            """
        )
        lines.append("--- pending evaluation (applied) ---")
        if not pend:
            lines.append("(none)")
        else:
            for r in pend:
                lines.append(
                    f"  id={r['action_id']} ts={r['action_ts']} {r['bucket_key']} "
                    f"{r['parameter_name']} {r['old_value']}→{r['new_value']}"
                )

        rev = self.db.fetchall(
            """
            SELECT action_id, action_ts, bucket_key, evaluation_status, reason_text
            FROM adaptive_actions
            WHERE IFNULL(reverted,0) = 1
            ORDER BY action_ts DESC
            LIMIT 6
            """
        )
        lines.append("--- last reverted ---")
        if not rev:
            lines.append("(none)")
        else:
            for r in rev:
                lines.append(
                    f"  id={r['action_id']} ts={r['action_ts']} {r['bucket_key']} "
                    f"ev={r['evaluation_status']}"
                )

        evlast = self.db.fetchall(
            """
            SELECT action_id, action_ts, bucket_key, evaluation_status,
                   json_extract(metrics_json, '$.post_avg_net_pnl_pct') AS post,
                   json_extract(metrics_json, '$.pre_avg_net_pnl_pct') AS pre
            FROM adaptive_actions
            WHERE IFNULL(applied,0) = 1
              AND IFNULL(evaluation_status,'') IN ('improved','worsened_reverted','inconclusive')
            ORDER BY action_ts DESC
            LIMIT 6
            """
        )
        lines.append("--- last evaluation results ---")
        if not evlast:
            lines.append("(none)")
        else:
            for r in evlast:
                lines.append(
                    f"  id={r['action_id']} ts={r['action_ts']} {r['bucket_key']} "
                    f"ev={r['evaluation_status']} pre%={r['pre']} post%={r['post']}"
                )

        rows = self.db.fetchall(
            """
            SELECT action_ts, bucket_key, parameter_name, old_value, new_value,
                   evaluation_status, applied, reverted, reason_text, action_id
            FROM adaptive_actions
            ORDER BY action_ts DESC
            LIMIT 10
            """
        )
        lines.append("--- recent adaptive_actions ---")
        for r in rows:
            lines.append(
                f"  {r['action_ts']} id={r['action_id']} {r['bucket_key']} {r['parameter_name']} "
                f"{r['old_value']}→{r['new_value']} ev={r['evaluation_status']} "
                f"app={r['applied']} rev={r['reverted']}"
            )
        lines.append("=== End ===")
        return "\n".join(lines)

    def balance_status_text(self) -> str:
        bal = load_paper_balance(self.cfg.balance_state_path)
        if not bal:
            return "paper_balance_state.json not found"
        pnl_since = bal.equity_rub - bal.initial_balance_rub
        parts = [
            f"initial_balance_rub={bal.initial_balance_rub:.2f}",
            f"current_balance_rub={bal.current_balance_rub:.2f}",
            f"realized_gross_pnl_rub={bal.realized_gross_pnl_rub:.2f}",
            f"realized_commission_rub={bal.realized_commission_rub:.2f}",
            f"realized_net_pnl_rub={bal.realized_net_pnl_rub:.2f}",
            f"unrealized_pnl_rub={bal.unrealized_pnl_rub:.2f}",
            f"equity_rub={bal.equity_rub:.2f}",
            f"pnl_since_reset≈{pnl_since:.2f}",
        ]
        if bal.last_reset_at:
            parts.append(f"last_reset_at={bal.last_reset_at}")
        return "\n".join(parts)

    def adaptive_reset(self) -> str:
        if self.cfg.trading_mode != "paper":
            return "reset allowed only in TRADING_MODE=paper"
        base_ml = float(os.getenv("ML_PROB_THRESHOLD", "0.63"))
        st = AdaptiveRuntimeState.default_for_base(base_ml)
        self.store.save(st)
        insert_adaptive_action(
            self.db,
            adaptive_mode=self.cfg.adaptive_mode,
            scope="global",
            bucket_key="all",
            parameter_name="reset_runtime_state",
            old_value=None,
            new_value=None,
            sample_size=0,
            confidence_score=1.0,
            action_status="manual_reset",
            reason_text="/adaptive_reset: cleared bucket thresholds and blocks",
            metrics={"reset": True},
            applied=False,
            evaluation_status="n/a",
        )
        return "adaptive runtime state reset to defaults"

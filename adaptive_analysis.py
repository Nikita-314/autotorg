from __future__ import annotations

import json
import statistics
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from analytics import AnalyticsDB, SignalLogger


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_float(value: Any) -> Optional[float]:
    try:
        val = float(value)
        if val != val:
            return None
        return val
    except (TypeError, ValueError):
        return None


def _safe_json_loads(raw: Any) -> Dict[str, Any]:
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else {}
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}


def volume_ratio_bucket(value: Any) -> str:
    num = _safe_float(value)
    if num is None:
        return "unknown"
    if num < 1.0:
        return "<1"
    if num <= 2.0:
        return "1-2"
    return ">2"


def volatility_bucket(value: Any) -> str:
    num = _safe_float(value)
    if num is None:
        return "unknown"
    if num < 0.01:
        return "<1%"
    if num <= 0.02:
        return "1-2%"
    return ">2%"


def _median(values: List[float]) -> Optional[float]:
    return statistics.median(values) if values else None


@dataclass
class BucketStats:
    bucket: str
    sample_size: int
    avg_outcome_15m_pct: Optional[float]
    median_outcome_15m_pct: Optional[float]
    avg_realized_gross_pnl_pct: Optional[float]
    avg_realized_net_pnl_pct: Optional[float]
    pct_positive: Optional[float]
    pct_negative: Optional[float]
    pct_zero: Optional[float]
    near_zero_pct: Optional[float]
    close_reason_top: List[List[Any]] = field(default_factory=list)
    same_bar_skip_count: int = 0
    block_trend_avg_outcome_15m_pct: Optional[float] = None
    block_ml_avg_outcome_15m_pct: Optional[float] = None


@dataclass
class AdaptiveRecommendation:
    scope: str
    bucket: str
    action: str
    base_value: float
    new_value: float
    confidence: float
    sample_size: int
    reason_text: str
    metrics: Dict[str, Any]


@dataclass
class AdaptiveDecision:
    scope: str
    bucket: str
    mode: str
    base_threshold: float
    effective_threshold: float
    threshold_delta: float
    confidence: float
    sample_size: int
    action: str
    should_apply: bool
    changed_decision: bool
    reason_text: str
    metrics: Dict[str, Any]


class AdaptiveAnalysisEngine:
    """Observation + recommendation + controlled paper/shadow adaptation."""

    def __init__(
        self,
        analytics_db: AnalyticsDB,
        analytics_logger: SignalLogger,
        *,
        state_path: Path,
        mode: str,
        base_ml_threshold: float,
        lookback_days: int = 14,
        refresh_cycles: int = 5,
        min_observations: int = 5,
        negative_outcome_pct: float = -0.25,
        positive_outcome_pct: float = 0.05,
        max_threshold_delta_frac: float = 0.15,
    ) -> None:
        self.analytics_db = analytics_db
        self.analytics_logger = analytics_logger
        self.state_path = Path(state_path)
        self.mode = (mode or "off").strip().lower()
        self.base_ml_threshold = float(base_ml_threshold)
        self.lookback_days = max(1, int(lookback_days))
        self.refresh_cycles = max(1, int(refresh_cycles))
        self.min_observations = max(1, int(min_observations))
        self.negative_outcome_pct = float(negative_outcome_pct)
        self.positive_outcome_pct = float(positive_outcome_pct)
        self.max_threshold_delta_frac = max(0.0, float(max_threshold_delta_frac))
        self._state = self._load_state()
        self.last_report: Dict[str, Any] = dict(self._state.get("last_report", {}))

    def _load_state(self) -> Dict[str, Any]:
        if not self.state_path.exists():
            return {
                "updated_at": None,
                "mode": self.mode,
                "volume_ratio_modifiers": {},
                "last_report": {},
            }
        try:
            with self.state_path.open("r", encoding="utf-8") as file:
                payload = json.load(file)
            if not isinstance(payload, dict):
                raise ValueError("adaptive state payload must be dict")
            payload.setdefault("updated_at", None)
            payload.setdefault("mode", self.mode)
            payload.setdefault("volume_ratio_modifiers", {})
            payload.setdefault("last_report", {})
            return payload
        except Exception:
            return {
                "updated_at": None,
                "mode": self.mode,
                "volume_ratio_modifiers": {},
                "last_report": {},
            }

    def _save_state(self) -> None:
        payload = {
            "updated_at": _now_iso(),
            "mode": self.mode,
            "volume_ratio_modifiers": self._state.get("volume_ratio_modifiers", {}),
            "last_report": self.last_report,
        }
        with self.state_path.open("w", encoding="utf-8") as file:
            json.dump(payload, file, ensure_ascii=True, indent=2)
        self._state = payload

    def _lookback_since(self) -> str:
        cutoff = datetime.now(timezone.utc) - timedelta(days=self.lookback_days)
        return cutoff.isoformat()

    def _fetch_buy_rows(self, since_iso: str) -> List[Dict[str, Any]]:
        rows = self.analytics_db.fetchall(
            """
            SELECT
              s.signal_id,
              s.ticker,
              s.signal_ts,
              s.market_regime,
              s.reason_code,
              s.reason_text,
              s.feature_snapshot_json,
              o.outcome_15m_pct,
              d.reason_text AS close_reason_text,
              d.reason_code AS close_reason_code,
              d.details_json AS close_details_json
            FROM signals s
            LEFT JOIN signal_outcomes o ON o.signal_id = s.signal_id
            LEFT JOIN decision_logs d
              ON d.signal_id = s.signal_id
             AND d.decision_type = 'TRADE_CLOSE'
            WHERE s.signal_ts >= ?
              AND s.side = 'BUY'
              AND s.reason_code = 'ENTRY_OK'
              AND o.outcome_15m_pct IS NOT NULL
            ORDER BY s.signal_ts DESC
            """,
            (since_iso,),
        )
        out: List[Dict[str, Any]] = []
        for row in rows:
            feat = _safe_json_loads(row.get("feature_snapshot_json"))
            close_details = _safe_json_loads(row.get("close_details_json"))
            out.append(
                {
                    **row,
                    "feature_snapshot": feat,
                    "volume_ratio_bucket": volume_ratio_bucket(feat.get("volume_ratio")),
                    "volatility_bucket": volatility_bucket(feat.get("volatility_20")),
                    "hour_utc": self._hour_utc(row.get("signal_ts")),
                    "realized_gross_pnl_pct": _safe_float(close_details.get("gross_pnl_pct")),
                    "realized_net_pnl_pct": _safe_float(close_details.get("net_pnl_pct"))
                    if _safe_float(close_details.get("net_pnl_pct")) is not None
                    else _safe_float(close_details.get("pnl_pct")),
                    "realized_gross_pnl_rub": _safe_float(close_details.get("gross_pnl")),
                    "realized_net_pnl_rub": _safe_float(close_details.get("net_pnl"))
                    if _safe_float(close_details.get("net_pnl")) is not None
                    else _safe_float(close_details.get("pnl")),
                    "commission_rub": _safe_float(close_details.get("commission_rub")),
                    "close_reason": row.get("close_reason_code") or row.get("close_reason_text"),
                    "near_zero_close": self._is_near_zero_close(close_details),
                }
            )
        return out

    def _fetch_block_rows(self, since_iso: str, reason_code: str) -> List[Dict[str, Any]]:
        rows = self.analytics_db.fetchall(
            """
            SELECT
              s.signal_id,
              s.ticker,
              s.signal_ts,
              s.feature_snapshot_json,
              o.outcome_15m_pct
            FROM signals s
            LEFT JOIN signal_outcomes o ON o.signal_id = s.signal_id
            WHERE s.signal_ts >= ?
              AND s.reason_code = ?
              AND o.outcome_15m_pct IS NOT NULL
            ORDER BY s.signal_ts DESC
            """,
            (since_iso, reason_code),
        )
        out: List[Dict[str, Any]] = []
        for row in rows:
            feat = _safe_json_loads(row.get("feature_snapshot_json"))
            out.append(
                {
                    **row,
                    "feature_snapshot": feat,
                    "volume_ratio_bucket": volume_ratio_bucket(feat.get("volume_ratio")),
                    "volatility_bucket": volatility_bucket(feat.get("volatility_20")),
                    "hour_utc": self._hour_utc(row.get("signal_ts")),
                }
            )
        return out

    @staticmethod
    def _hour_utc(signal_ts: Any) -> Optional[int]:
        if not signal_ts:
            return None
        try:
            return datetime.fromisoformat(str(signal_ts)).hour
        except ValueError:
            return None

    @staticmethod
    def _is_near_zero_close(close_details: Dict[str, Any]) -> bool:
        pnl = (
            _safe_float(close_details.get("net_pnl"))
            if _safe_float(close_details.get("net_pnl")) is not None
            else _safe_float(close_details.get("pnl"))
        )
        pnl_pct = (
            _safe_float(close_details.get("net_pnl_pct"))
            if _safe_float(close_details.get("net_pnl_pct")) is not None
            else _safe_float(close_details.get("pnl_pct"))
        )
        if pnl is not None and abs(pnl) < 10:
            return True
        if pnl_pct is not None and abs(pnl_pct) < 0.05:
            return True
        return False

    def _same_bar_skip_count(self, since_iso: str) -> int:
        row = self.analytics_db.fetchone(
            """
            SELECT COUNT(*) AS cnt
            FROM decision_logs
            WHERE decision_ts >= ?
              AND reason_code = 'SAME_BAR_REENTRY'
            """,
            (since_iso,),
        )
        return int(row["cnt"]) if row else 0

    def _build_bucket_stats(
        self,
        rows: List[Dict[str, Any]],
        key: str,
        block_trend_rows: List[Dict[str, Any]],
        block_ml_rows: List[Dict[str, Any]],
        same_bar_skip_count: int,
    ) -> Dict[str, BucketStats]:
        grouped: Dict[str, List[Dict[str, Any]]] = {}
        for row in rows:
            bucket = str(row.get(key, "unknown"))
            grouped.setdefault(bucket, []).append(row)

        block_trend_grouped: Dict[str, List[float]] = {}
        for row in block_trend_rows:
            bucket = str(row.get(key, "unknown"))
            outcome = _safe_float(row.get("outcome_15m_pct"))
            if outcome is not None:
                block_trend_grouped.setdefault(bucket, []).append(outcome)

        block_ml_grouped: Dict[str, List[float]] = {}
        for row in block_ml_rows:
            bucket = str(row.get(key, "unknown"))
            outcome = _safe_float(row.get("outcome_15m_pct"))
            if outcome is not None:
                block_ml_grouped.setdefault(bucket, []).append(outcome)

        stats_by_bucket: Dict[str, BucketStats] = {}
        for bucket, bucket_rows in grouped.items():
            outcomes = [_safe_float(row.get("outcome_15m_pct")) for row in bucket_rows]
            outcome_vals = [v for v in outcomes if v is not None]
            realized_gross_vals = [
                _safe_float(row.get("realized_gross_pnl_pct"))
                for row in bucket_rows
                if _safe_float(row.get("realized_gross_pnl_pct")) is not None
            ]
            realized_net_vals = [
                _safe_float(row.get("realized_net_pnl_pct"))
                for row in bucket_rows
                if _safe_float(row.get("realized_net_pnl_pct")) is not None
            ]
            positive = sum(1 for v in outcome_vals if v > 0)
            negative = sum(1 for v in outcome_vals if v < 0)
            zero = sum(1 for v in outcome_vals if v == 0)
            close_reasons: Dict[str, int] = {}
            near_zero_count = 0
            for row in bucket_rows:
                if row.get("close_reason"):
                    reason_key = str(row["close_reason"])
                    close_reasons[reason_key] = close_reasons.get(reason_key, 0) + 1
                if row.get("near_zero_close"):
                    near_zero_count += 1
            sorted_reasons = sorted(close_reasons.items(), key=lambda item: (-item[1], item[0]))[:3]
            n = len(outcome_vals)
            stats_by_bucket[bucket] = BucketStats(
                bucket=bucket,
                sample_size=n,
                avg_outcome_15m_pct=(sum(outcome_vals) / n) if n else None,
                median_outcome_15m_pct=_median(outcome_vals),
                avg_realized_gross_pnl_pct=(
                    (sum(realized_gross_vals) / len(realized_gross_vals)) if realized_gross_vals else None
                ),
                avg_realized_net_pnl_pct=(
                    (sum(realized_net_vals) / len(realized_net_vals)) if realized_net_vals else None
                ),
                pct_positive=(100.0 * positive / n) if n else None,
                pct_negative=(100.0 * negative / n) if n else None,
                pct_zero=(100.0 * zero / n) if n else None,
                near_zero_pct=(100.0 * near_zero_count / n) if n else None,
                close_reason_top=[[k, v] for k, v in sorted_reasons],
                same_bar_skip_count=same_bar_skip_count,
                block_trend_avg_outcome_15m_pct=(
                    sum(block_trend_grouped.get(bucket, [])) / len(block_trend_grouped[bucket])
                    if block_trend_grouped.get(bucket)
                    else None
                ),
                block_ml_avg_outcome_15m_pct=(
                    sum(block_ml_grouped.get(bucket, [])) / len(block_ml_grouped[bucket])
                    if block_ml_grouped.get(bucket)
                    else None
                ),
            )
        return stats_by_bucket

    def analyze(self) -> Dict[str, Any]:
        since_iso = self._lookback_since()
        buy_rows = self._fetch_buy_rows(since_iso)
        block_trend_rows = self._fetch_block_rows(since_iso, "BLOCK_TREND")
        block_ml_rows = self._fetch_block_rows(since_iso, "BLOCK_ML")
        same_bar_skip_count = self._same_bar_skip_count(since_iso)

        report = {
            "generated_at": _now_iso(),
            "since_iso": since_iso,
            "buy_count": len(buy_rows),
            "block_trend_count": len(block_trend_rows),
            "block_ml_count": len(block_ml_rows),
            "same_bar_skip_count": same_bar_skip_count,
            "by_volume_ratio": {
                bucket: asdict(stats)
                for bucket, stats in self._build_bucket_stats(
                    buy_rows, "volume_ratio_bucket", block_trend_rows, block_ml_rows, same_bar_skip_count
                ).items()
            },
            "by_hour_utc": {
                str(bucket): asdict(stats)
                for bucket, stats in self._build_bucket_stats(
                    buy_rows, "hour_utc", block_trend_rows, block_ml_rows, same_bar_skip_count
                ).items()
            },
            "by_ticker": {
                bucket: asdict(stats)
                for bucket, stats in self._build_bucket_stats(
                    buy_rows, "ticker", block_trend_rows, block_ml_rows, same_bar_skip_count
                ).items()
            },
            "top_good_buy": [
                {
                    "ticker": row["ticker"],
                    "signal_ts": row["signal_ts"],
                    "outcome_15m_pct": row["outcome_15m_pct"],
                    "realized_net_pnl_pct": row["realized_net_pnl_pct"],
                    "volume_ratio_bucket": row["volume_ratio_bucket"],
                    "hour_utc": row["hour_utc"],
                }
                for row in sorted(
                    buy_rows, key=lambda item: (_safe_float(item.get("outcome_15m_pct")) or float("-inf")), reverse=True
                )[:5]
            ],
            "top_bad_buy": [
                {
                    "ticker": row["ticker"],
                    "signal_ts": row["signal_ts"],
                    "outcome_15m_pct": row["outcome_15m_pct"],
                    "realized_net_pnl_pct": row["realized_net_pnl_pct"],
                    "volume_ratio_bucket": row["volume_ratio_bucket"],
                    "hour_utc": row["hour_utc"],
                }
                for row in sorted(
                    buy_rows, key=lambda item: (_safe_float(item.get("outcome_15m_pct")) or float("inf"))
                )[:5]
            ],
        }
        self.last_report = report
        return report

    def _recommend_from_bucket(self, bucket: BucketStats) -> Optional[AdaptiveRecommendation]:
        primary_metric = (
            bucket.avg_realized_net_pnl_pct
            if bucket.avg_realized_net_pnl_pct is not None
            else bucket.avg_outcome_15m_pct
        )
        primary_metric_name = "avg_realized_net_pnl_pct" if bucket.avg_realized_net_pnl_pct is not None else "avg_outcome_15m_pct"
        if bucket.sample_size < self.min_observations or primary_metric is None:
            return None

        max_abs_delta = self.base_ml_threshold * self.max_threshold_delta_frac
        if max_abs_delta <= 0:
            return None

        magnitude = abs(primary_metric)
        sample_strength = min(1.0, bucket.sample_size / max(self.min_observations * 2, 1))
        outcome_strength = min(1.0, magnitude / 1.0)
        confidence = max(0.05, min(0.99, 0.45 * sample_strength + 0.55 * outcome_strength))

        if primary_metric <= self.negative_outcome_pct:
            delta = max_abs_delta * confidence
            new_value = min(0.99, self.base_ml_threshold + delta)
            reason_text = (
                f"Raised ML threshold for volume_ratio {bucket.bucket} from "
                f"{self.base_ml_threshold:.4f} to {new_value:.4f} because last "
                f"{bucket.sample_size} BUY had {primary_metric_name}={primary_metric:.4f} "
                f"and near_zero_pct={bucket.near_zero_pct or 0.0:.2f}."
            )
            return AdaptiveRecommendation(
                scope="volume_ratio",
                bucket=bucket.bucket,
                action="raise_threshold",
                base_value=self.base_ml_threshold,
                new_value=new_value,
                confidence=confidence,
                sample_size=bucket.sample_size,
                reason_text=reason_text,
                metrics=asdict(bucket),
            )

        if primary_metric >= self.positive_outcome_pct:
            delta = max_abs_delta * confidence * 0.7
            new_value = max(0.0, self.base_ml_threshold - delta)
            reason_text = (
                f"Lowered ML threshold for volume_ratio {bucket.bucket} from "
                f"{self.base_ml_threshold:.4f} to {new_value:.4f} because last "
                f"{bucket.sample_size} BUY had {primary_metric_name}={primary_metric:.4f} "
                f"and pct_positive={bucket.pct_positive or 0.0:.2f}."
            )
            return AdaptiveRecommendation(
                scope="volume_ratio",
                bucket=bucket.bucket,
                action="lower_threshold",
                base_value=self.base_ml_threshold,
                new_value=new_value,
                confidence=confidence,
                sample_size=bucket.sample_size,
                reason_text=reason_text,
                metrics=asdict(bucket),
            )
        return None

    def refresh(self, cycle: int) -> None:
        if cycle > 1 and (cycle % self.refresh_cycles) != 0:
            return

        report = self.analyze()
        old_modifiers = dict(self._state.get("volume_ratio_modifiers", {}))
        new_modifiers: Dict[str, Dict[str, Any]] = {}
        by_volume = report.get("by_volume_ratio", {})

        for raw_bucket, payload in by_volume.items():
            stats = BucketStats(**payload)
            recommendation = self._recommend_from_bucket(stats)
            if recommendation is None:
                continue
            new_modifiers[raw_bucket] = {
                "threshold_delta": recommendation.new_value - recommendation.base_value,
                "effective_threshold": recommendation.new_value,
                "confidence": recommendation.confidence,
                "sample_size": recommendation.sample_size,
                "action": recommendation.action,
                "reason_text": recommendation.reason_text,
                "metrics": recommendation.metrics,
                "updated_at": _now_iso(),
            }

        self._state["volume_ratio_modifiers"] = new_modifiers
        self._state["mode"] = self.mode

        for bucket in sorted(set(old_modifiers) | set(new_modifiers)):
            old_value = self.base_ml_threshold + _safe_float(old_modifiers.get(bucket, {}).get("threshold_delta") or 0.0)
            new_value = self.base_ml_threshold + _safe_float(new_modifiers.get(bucket, {}).get("threshold_delta") or 0.0)
            if abs(new_value - old_value) < 1e-9:
                continue
            modifier = new_modifiers.get(bucket, {})
            status = "OFF"
            if self.mode == "shadow" and modifier:
                status = "SHADOW_RECOMMENDATION"
            elif self.mode == "paper" and modifier:
                status = "APPLIED"
            elif not modifier:
                status = "CLEARED"
            self.analytics_logger.log_adaptation_action(
                mode=self.mode,
                scope="volume_ratio",
                bucket_key=bucket,
                parameter_name="ml_prob_threshold",
                old_value=old_value,
                new_value=new_value,
                confidence_score=_safe_float(modifier.get("confidence")),
                sample_size=int(modifier.get("sample_size", 0) or 0),
                action_status=status,
                reason_text=str(modifier.get("reason_text") or "Adaptive modifier cleared."),
                metrics=modifier.get("metrics", {}),
                window_summary={
                    "lookback_days": self.lookback_days,
                    "since_iso": report.get("since_iso"),
                    "buy_count": report.get("buy_count"),
                    "block_trend_count": report.get("block_trend_count"),
                    "block_ml_count": report.get("block_ml_count"),
                    "same_bar_skip_count": report.get("same_bar_skip_count"),
                },
            )

        self._save_state()

    def evaluate_entry(
        self,
        *,
        symbol: str,
        feature_snapshot: Dict[str, Any],
        market_regime: str,
        ml_prob_up: Optional[float],
        trading_mode: str,
    ) -> Optional[AdaptiveDecision]:
        if self.mode not in ("shadow", "paper"):
            return None
        if ml_prob_up is None:
            return None

        bucket = volume_ratio_bucket(feature_snapshot.get("volume_ratio"))
        modifier = self._state.get("volume_ratio_modifiers", {}).get(bucket)
        if not modifier:
            return None

        delta = _safe_float(modifier.get("threshold_delta")) or 0.0
        effective_threshold = min(0.99, max(0.0, self.base_ml_threshold + delta))
        if abs(effective_threshold - self.base_ml_threshold) < 1e-9:
            return None

        base_allows = ml_prob_up >= self.base_ml_threshold
        adapted_allows = ml_prob_up >= effective_threshold
        changed_decision = base_allows != adapted_allows
        should_apply = self.mode == "paper" and trading_mode == "paper"
        action = str(modifier.get("action") or "adjust_threshold")
        confidence = _safe_float(modifier.get("confidence")) or 0.0
        sample_size = int(modifier.get("sample_size", 0) or 0)
        reason_text = str(modifier.get("reason_text") or "Adaptive regime filter active.")
        if self.mode == "shadow":
            reason_text = f"[shadow] {reason_text}"

        return AdaptiveDecision(
            scope="volume_ratio",
            bucket=bucket,
            mode=self.mode,
            base_threshold=self.base_ml_threshold,
            effective_threshold=effective_threshold,
            threshold_delta=delta,
            confidence=confidence,
            sample_size=sample_size,
            action=action,
            should_apply=should_apply,
            changed_decision=changed_decision,
            reason_text=reason_text,
            metrics={
                "symbol": symbol,
                "market_regime": market_regime,
                "volume_ratio": feature_snapshot.get("volume_ratio"),
                "volatility_20": feature_snapshot.get("volatility_20"),
                "ml_prob_up": ml_prob_up,
                "modifier_metrics": modifier.get("metrics", {}),
            },
        )

    def log_signal_adaptation(
        self,
        *,
        signal_id: Optional[str],
        run_id: Optional[str],
        ticker: str,
        decision: AdaptiveDecision,
    ) -> None:
        if not decision.changed_decision and self.mode == "off":
            return
        self.analytics_logger.log_decision(
            run_id=run_id,
            signal_id=signal_id,
            ticker=ticker,
            decision_type="ADAPTATION",
            decision_label="APPLY" if decision.should_apply and decision.changed_decision else "SHADOW",
            reason_code=f"ADAPTIVE_{decision.scope.upper()}",
            reason_text=decision.reason_text,
            details={
                "scope": decision.scope,
                "bucket": decision.bucket,
                "mode": decision.mode,
                "base_threshold": decision.base_threshold,
                "effective_threshold": decision.effective_threshold,
                "threshold_delta": decision.threshold_delta,
                "confidence": decision.confidence,
                "sample_size": decision.sample_size,
                "changed_decision": decision.changed_decision,
                "metrics": decision.metrics,
            },
        )

    def describe_policy(self) -> str:
        modifiers = self._state.get("volume_ratio_modifiers", {})
        if not modifiers:
            return f"adaptive={self.mode} | no active regime modifiers"
        parts = []
        for bucket in sorted(modifiers):
            eff = _safe_float(modifiers[bucket].get("effective_threshold"))
            conf = _safe_float(modifiers[bucket].get("confidence"))
            if eff is None:
                continue
            parts.append(f"{bucket}->{eff:.3f} (conf={conf or 0.0:.2f})")
        return f"adaptive={self.mode} | volume_ratio modifiers: " + ", ".join(parts)

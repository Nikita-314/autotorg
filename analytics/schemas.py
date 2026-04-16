from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class SignalPayload:
    signal_id: str
    run_id: Optional[str]
    ticker: str
    signal_ts: str
    strategy_code: str
    strategy_version: str
    side: str
    entry_price: Optional[float]
    stop_price: Optional[float]
    target_price: Optional[float]
    confidence_score: Optional[float]
    reason_code: Optional[str]
    reason_text: Optional[str]
    feature_snapshot_json: Dict[str, Any] = field(default_factory=dict)
    model_snapshot_json: Dict[str, Any] = field(default_factory=dict)
    market_regime: Optional[str] = None
    execution_mode: Optional[str] = None
    status: str = "NEW"


@dataclass
class DecisionPayload:
    decision_id: str
    run_id: Optional[str]
    signal_id: Optional[str]
    ticker: str
    decision_ts: str
    decision_type: str
    decision_label: str
    reason_code: Optional[str] = None
    reason_text: Optional[str] = None
    details_json: Dict[str, Any] = field(default_factory=dict)


@dataclass
class InferencePayload:
    inference_id: str
    signal_id: Optional[str]
    run_id: Optional[str]
    ticker: Optional[str]
    model_type: str
    model_version: Optional[str]
    inference_ts: str
    input_features_json: Dict[str, Any] = field(default_factory=dict)
    raw_output_json: Dict[str, Any] = field(default_factory=dict)
    decision_label: Optional[str] = None
    confidence_score: Optional[float] = None
    action_recommendation: Optional[str] = None
    model_used_in_final_decision: bool = False


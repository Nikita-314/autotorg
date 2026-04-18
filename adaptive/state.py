from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from .buckets import BucketKey


@dataclass
class AdaptiveRuntimeState:
    version: int = 1
    base_ml_threshold: float = 0.63
    # Абсолютный эффективный порог по bucket (уже с clamp); None = нет данных
    effective_threshold: Dict[str, float] = field(
        default_factory=lambda: {"lt1": 0.63, "b12": 0.63, "gt2": 0.63}
    )
    blocked: Dict[str, bool] = field(
        default_factory=lambda: {"lt1": False, "b12": False, "gt2": False}
    )
    updated_at: str = ""

    @staticmethod
    def clamp_threshold(base: float, value: float, max_rel: float) -> float:
        lo = base * (1.0 - max_rel)
        hi = base * (1.0 + max_rel)
        return max(lo, min(hi, value))

    @staticmethod
    def default_for_base(base: float) -> "AdaptiveRuntimeState":
        b = float(base)
        return AdaptiveRuntimeState(
            base_ml_threshold=b,
            effective_threshold={"lt1": b, "b12": b, "gt2": b},
            blocked={"lt1": False, "b12": False, "gt2": False},
            updated_at=datetime.now(timezone.utc).isoformat(),
        )

    def to_json(self) -> Dict[str, Any]:
        d = asdict(self)
        return d

    @staticmethod
    def from_json(data: Dict[str, Any]) -> "AdaptiveRuntimeState":
        return AdaptiveRuntimeState(
            version=int(data.get("version", 1)),
            base_ml_threshold=float(data.get("base_ml_threshold", 0.63)),
            effective_threshold={
                k: float(v)
                for k, v in (data.get("effective_threshold") or {}).items()
            },
            blocked={k: bool(v) for k, v in (data.get("blocked") or {}).items()},
            updated_at=str(data.get("updated_at") or ""),
        )


class AdaptiveStateStore:
    def __init__(self, path: Path):
        self.path = Path(path)

    def load(self, base_ml_threshold: float) -> AdaptiveRuntimeState:
        if not self.path.exists():
            st = AdaptiveRuntimeState.default_for_base(base_ml_threshold)
            self.save(st)
            return st
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            st = AdaptiveRuntimeState.from_json(raw)
            # Синхронизируем базу из env/модели при смене конфига
            if abs(st.base_ml_threshold - base_ml_threshold) > 1e-9:
                st.base_ml_threshold = float(base_ml_threshold)
            for k in ("lt1", "b12", "gt2"):
                st.effective_threshold.setdefault(k, st.base_ml_threshold)
                st.blocked.setdefault(k, False)
            return st
        except Exception:
            st = AdaptiveRuntimeState.default_for_base(base_ml_threshold)
            self.save(st)
            return st

    def save(self, state: AdaptiveRuntimeState) -> None:
        state.updated_at = datetime.now(timezone.utc).isoformat()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(state.to_json(), indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(self.path)

    def set_effective(self, state: AdaptiveRuntimeState, bucket: BucketKey, value: float, max_rel: float) -> float:
        clamped = AdaptiveRuntimeState.clamp_threshold(state.base_ml_threshold, value, max_rel)
        state.effective_threshold[bucket] = clamped
        return clamped

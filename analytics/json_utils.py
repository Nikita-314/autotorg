"""JSON-friendly conversion for snapshots (numpy/pandas scalars, NaN)."""
from __future__ import annotations

import math
from decimal import Decimal
from typing import Any, Dict, List, Union

_JSONScalar = Union[str, int, float, bool, None]
JSONValue = Union[_JSONScalar, Dict[str, Any], List[Any]]


def safe_to_json(obj: Any) -> JSONValue:
    if obj is None:
        return None
    if isinstance(obj, dict):
        return {str(k): safe_to_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [safe_to_json(v) for v in obj]
    if isinstance(obj, bool):
        return obj
    if isinstance(obj, int) and not isinstance(obj, bool):
        return int(obj)
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return float(obj)
    if isinstance(obj, Decimal):
        try:
            x = float(obj)
        except Exception:
            return None
        if math.isnan(x) or math.isinf(x):
            return None
        return x
    try:
        import numpy as np  # type: ignore

        if isinstance(obj, (np.floating, np.float32, np.float64)):
            x = float(obj)
            return None if math.isnan(x) or math.isinf(x) else x
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return safe_to_json(obj.tolist())
    except Exception:
        pass
    try:
        import pandas as pd  # type: ignore

        if isinstance(obj, pd.Timestamp):
            return obj.isoformat()
    except Exception:
        pass
    if isinstance(obj, str):
        return obj
    try:
        return float(obj)
    except Exception:
        try:
            return str(obj)
        except Exception:
            return None

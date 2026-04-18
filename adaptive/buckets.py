from __future__ import annotations

from typing import Literal, Optional

BucketKey = Literal["lt1", "b12", "gt2"]

BUCKET_LABELS = {"lt1": "<1", "b12": "1-2", "gt2": ">2"}


def volume_ratio_to_bucket(volume_ratio: Optional[float]) -> Optional[BucketKey]:
    if volume_ratio is None:
        return None
    try:
        v = float(volume_ratio)
    except (TypeError, ValueError):
        return None
    if v < 1.0:
        return "lt1"
    if v <= 2.0:
        return "b12"
    return "gt2"

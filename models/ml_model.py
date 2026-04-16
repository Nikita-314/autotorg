"""
ML-классификатор: предсказывает P(up) — вероятность роста через N свечей.
Используется вместе с Supertrend как фильтр качества входа.
"""
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

from features import build_features, make_target

LOGGER = logging.getLogger("ml-model")

FEATURE_COLS = [
    "atr_pct", "dist_to_st", "close_above_st", "ema_dist_50", "ema_dist_200",
    "rsi", "macd", "macd_hist", "ret_1", "ret_3", "ret_5", "ret_10",
    "volatility", "vol_ratio", "hour", "dow",
]


class MLSignalModel:
    """
    Supertrend + ML: Supertrend задаёт направление, ML — качество входа.
    Вход long только если: st_dir=1, close>ema200, P(up)>threshold.
    """

    def __init__(
        self,
        model_path: Optional[Path] = None,
        scaler_path: Optional[Path] = None,
        prob_threshold: float = 0.63,
        forward_bars: int = 10,
        target_threshold_pct: float = 0.008,
    ):
        base = Path(__file__).resolve().parent.parent
        self.model_path = model_path or base / "ml_model.joblib"
        self.scaler_path = scaler_path or base / "ml_scaler.pkl"
        self.prob_threshold = prob_threshold
        self.forward_bars = forward_bars
        self.target_threshold_pct = target_threshold_pct
        self.model = None
        self.scaler = StandardScaler()
        self.feature_cols = FEATURE_COLS
        self._is_fitted = False

    def _prepare_xy(self, df: pd.DataFrame) -> tuple:
        """Features + target, без NaN."""
        features = build_features(df)
        features["target"] = make_target(
            df["close"],
            forward_bars=self.forward_bars,
            threshold_pct=self.target_threshold_pct,
        )
        features = features.dropna()
        if len(features) < 100:
            return None, None
        X = features[self.feature_cols].copy()
        y = features["target"]
        return X, y

    def fit(self, symbols_data: Dict[str, pd.DataFrame]) -> Dict[str, float]:
        """
        Обучает модель на данных нескольких символов.
        symbols_data: {symbol: DataFrame with OHLCV}
        Возвращает метрики.
        """
        all_X: List[pd.DataFrame] = []
        all_y: List[pd.Series] = []
        for symbol, df in symbols_data.items():
            if df is None or len(df) < 100:
                continue
            X, y = self._prepare_xy(df)
            if X is not None and y is not None:
                all_X.append(X)
                all_y.append(y)
        if not all_X:
            LOGGER.warning("No data for ML training")
            return {"error": "no_data"}
        X = pd.concat(all_X, axis=0, ignore_index=True)
        y = pd.concat(all_y, axis=0, ignore_index=True)
        mask = X.notna().all(axis=1) & y.notna()
        X = X[mask].dropna()
        y = y[mask].dropna()
        if len(X) != len(y):
            n = min(len(X), len(y))
            X = X.iloc[:n]
            y = y.iloc[:n]
        if len(X) < 200:
            LOGGER.warning("Too few samples for ML: %d", len(X))
            return {"error": "too_few_samples", "n": len(X)}

        self.scaler.fit(X)
        X_scaled = self.scaler.transform(X)
        base = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42)
        self.model = CalibratedClassifierCV(base, cv=3, method="isotonic")
        self.model.fit(X_scaled, y.values)
        self._is_fitted = True

        # Walk-forward style eval
        tscv = TimeSeriesSplit(n_splits=3)
        scores = []
        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            X_train_s = self.scaler.transform(X_train)
            X_test_s = self.scaler.transform(X_test)
            m = CalibratedClassifierCV(
                RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42),
                cv=2, method="isotonic",
            )
            m.fit(X_train_s, y_train.values)
            proba = m.predict_proba(X_test_s)[:, 1]
            acc = (proba > 0.5).astype(int) == y_test.values
            scores.append(acc.mean())
        metrics = {"cv_accuracy": float(np.mean(scores)), "n_samples": len(X)}
        LOGGER.info("ML model fitted: cv_accuracy=%.3f n=%d", metrics["cv_accuracy"], metrics["n_samples"])
        return metrics

    def predict_proba_up(self, features: pd.DataFrame) -> Optional[float]:
        """Вероятность роста. Возвращает None если модель не обучена или нет данных."""
        if not self._is_fitted or self.model is None:
            return None
        available = [c for c in self.feature_cols if c in features.columns]
        if len(available) < len(self.feature_cols) // 2:
            return None
        row = features[available].iloc[-1:].copy()
        row = row.replace([np.inf, -np.inf], np.nan).fillna(0)
        for c in self.feature_cols:
            if c not in row.columns:
                row[c] = 0
        row = row[self.feature_cols]
        try:
            X = self.scaler.transform(row)
            proba = self.model.predict_proba(X)[0, 1]
            return float(proba)
        except Exception:  # pylint: disable=broad-except
            return None

    def save(self) -> None:
        """Сохраняет модель и scaler."""
        if not self._is_fitted:
            return
        try:
            import joblib
            joblib.dump(self.model, self.model_path)
            joblib.dump(self.scaler, self.scaler_path)
            LOGGER.info("ML model saved to %s", self.model_path)
        except Exception as exc:  # pylint: disable=broad-except
            LOGGER.warning("Cannot save ML model: %s", exc)

    def load(self) -> bool:
        """Загружает модель и scaler."""
        try:
            import joblib
            if self.model_path.exists():
                self.model = joblib.load(self.model_path)
                self.scaler = joblib.load(self.scaler_path)
                self._is_fitted = True
                LOGGER.info("ML model loaded from %s", self.model_path)
                return True
        except Exception as exc:  # pylint: disable=broad-except
            LOGGER.warning("Cannot load ML model: %s", exc)
        return False

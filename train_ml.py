#!/usr/bin/env python3
"""
Обучение ML-модели на исторических данных.
Запуск: python train_ml.py
Соберёт данные по символам из TV_SYMBOLS или MOEX, обучит модель, сохранит ml_model.joblib.
"""
import logging
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
LOGGER = logging.getLogger("train_ml")

from data_loader import load_candles
from models.ml_model import MLSignalModel


def main() -> None:
    exchange = os.getenv("TV_EXCHANGE", "MOEX").strip()
    interval = os.getenv("TV_INTERVAL", "1h").strip()
    symbols_str = os.getenv("TV_SYMBOLS", "SBER,GAZP,LKOH,ROSN,NVTK,AFLT,GMKN,NORNL")
    symbols = [s.strip().upper() for s in symbols_str.split(",") if s.strip()]

    LOGGER.info("Loading data for %s on %s, interval=%s", symbols, exchange, interval)
    data: dict = {}
    for sym in symbols:
        df = load_candles(sym, exchange, interval)
        if df is not None and len(df) >= 100:
            data[sym] = df
            LOGGER.info("  %s: %d rows", sym, len(df))
        else:
            LOGGER.warning("  %s: skip (no data or too few)", sym)

    if len(data) < 1:
        LOGGER.error("No data for training")
        return

    model = MLSignalModel(
        model_path=Path("ml_model.joblib"),
        scaler_path=Path("ml_scaler.pkl"),
        prob_threshold=float(os.getenv("ML_PROB_THRESHOLD", "0.63")),
    )
    metrics = model.fit(data)
    if "error" in metrics:
        LOGGER.error("Training failed: %s", metrics)
        return
    model.save()
    LOGGER.info("Model saved. cv_accuracy=%.3f n=%d", metrics.get("cv_accuracy", 0), metrics.get("n_samples", 0))


if __name__ == "__main__":
    main()

-- Показывает результативность торговых сигналов (BUY/SELL) по часу возникновения сигнала.
-- Интерпретация: позволяет понять, в какие часы стратегия статистически сильнее или слабее.
-- Примечание: на текущем этапе signal_ts пишется как timezone-aware ISO timestamp (+00:00),
-- но для mixed historical data поле часа оставлено нейтральным (без жесткой привязки в названии).
SELECT
  CAST(strftime('%H', s.signal_ts) AS INTEGER) AS signal_hour,
  COUNT(*) AS signals_total,
  AVG(o.outcome_15m_pct) AS avg_15m_pct,
  AVG(o.outcome_60m_pct) AS avg_60m_pct,
  AVG(o.outcome_eod_pct) AS avg_eod_pct
FROM signals s
JOIN signal_outcomes o ON o.signal_id = s.signal_id
WHERE s.side IN ('BUY', 'SELL')
GROUP BY CAST(strftime('%H', s.signal_ts) AS INTEGER)
ORDER BY signal_hour;

-- Показывает среднюю доходность по торговым сигналам (только BUY/SELL; HOLD/BLOCK исключены).
-- Интерпретация: положительные значения выше нуля означают, что торговые сигналы в среднем отрабатывают в плюс.
SELECT
  s.strategy_code,
  COUNT(*) AS signals_total,
  AVG(o.outcome_5m_pct) AS avg_5m_pct,
  AVG(o.outcome_15m_pct) AS avg_15m_pct,
  AVG(o.outcome_60m_pct) AS avg_60m_pct,
  AVG(o.outcome_eod_pct) AS avg_eod_pct
FROM signals s
JOIN signal_outcomes o ON o.signal_id = s.signal_id
WHERE s.side IN ('BUY', 'SELL')
GROUP BY s.strategy_code
ORDER BY avg_eod_pct DESC;

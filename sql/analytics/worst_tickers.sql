-- Показывает худшие тикеры по среднему EOD-результату для торговых сигналов (BUY/SELL).
-- Интерпретация: кандидаты для исключения из universe или отдельного risk-фильтра.
SELECT
  s.ticker,
  COUNT(*) AS signals_total,
  AVG(o.outcome_eod_pct) AS avg_eod_pct,
  MIN(o.outcome_eod_pct) AS worst_eod_pct,
  AVG(o.mae) AS avg_mae
FROM signals s
JOIN signal_outcomes o ON o.signal_id = s.signal_id
WHERE s.side IN ('BUY', 'SELL')
GROUP BY s.ticker
HAVING COUNT(*) >= 3
ORDER BY avg_eod_pct ASC
LIMIT 20;

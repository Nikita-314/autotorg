-- Показывает ML-блокировки и их качество: что было бы, если бы вход не блокировался.
-- Интерпретация: если при BLOCK_ML средний outcome положительный, ML может душить хорошие сигналы.
-- Ограничение: outcome для BLOCK_ML появляется только после работы outcome evaluator и при наличии баров после события.
SELECT
  DATE(s.signal_ts) AS day,
  COUNT(*) AS blocked_count,
  AVG(CAST(json_extract(s.model_snapshot_json, '$.ml_prob_up') AS REAL)) AS avg_ml_prob,
  AVG(o.outcome_15m_pct) AS avg_blocked_outcome_15m,
  AVG(o.outcome_60m_pct) AS avg_blocked_outcome_60m,
  AVG(o.outcome_eod_pct) AS avg_blocked_outcome_eod
FROM signals s
LEFT JOIN signal_outcomes o ON o.signal_id = s.signal_id
WHERE s.side IN ('HOLD', 'BLOCK')
  AND s.reason_code = 'BLOCK_ML'
GROUP BY DATE(s.signal_ts)
ORDER BY day DESC;

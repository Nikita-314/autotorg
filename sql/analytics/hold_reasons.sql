-- Показывает основные причины HOLD/BLOCK решений.
-- Интерпретация: если растет доля BLOCK_ML или BLOCK_TREND, это «узкое горлышко» текущего фильтра.
SELECT
  COALESCE(reason_code, 'UNKNOWN') AS reason_code,
  decision_label,
  COUNT(*) AS total
FROM decision_logs
WHERE decision_type = 'STRATEGY_DECISION'
  AND decision_label IN ('HOLD', 'BLOCK')
GROUP BY COALESCE(reason_code, 'UNKNOWN'), decision_label
ORDER BY total DESC;

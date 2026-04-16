-- SAME_BAR_REENTRY analytics (decision_logs.details_json содержит bar_key, interval)
-- sqlite3 analytics.db < analytics/same_bar_reentry_report.sql

SELECT 'total' AS section, COUNT(*) AS n
FROM decision_logs
WHERE reason_code = 'SAME_BAR_REENTRY';

SELECT 'by_ticker' AS section, ticker, COUNT(*) AS cnt
FROM decision_logs
WHERE reason_code = 'SAME_BAR_REENTRY'
GROUP BY ticker
ORDER BY cnt DESC, ticker;

SELECT
  'by_hour_utc' AS section,
  CASE
    WHEN instr(decision_ts, 'T') > 0
    THEN substr(decision_ts, instr(decision_ts, 'T') + 1, 2)
    ELSE NULL
  END AS hour_utc,
  COUNT(*) AS cnt
FROM decision_logs
WHERE reason_code = 'SAME_BAR_REENTRY'
GROUP BY hour_utc
ORDER BY hour_utc;

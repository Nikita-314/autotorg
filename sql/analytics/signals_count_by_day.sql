-- Показывает количество всех типов сигналов по дням: BUY/SELL (торговые) и HOLD/BLOCK (диагностические).
-- Интерпретация: помогает увидеть, когда бот «замолкает», переактивен или слишком часто блокирует входы.
SELECT
  DATE(signal_ts) AS day,
  side,
  COUNT(*) AS signals_count
FROM signals
GROUP BY DATE(signal_ts), side
ORDER BY day DESC, side;

#!/bin/bash
# Запуск бота в screen с выводом в консоль (без >> bot.log 2>&1)
# Логи пишутся в bot.log через BOT_LOG_FILE в .env
cd "$(dirname "$0")"
. .venv/bin/activate
export ANALYTICS_ENABLED=true
export ANALYTICS_DB_PATH=analytics.db
export ANALYTICS_LOG_HOLDS=true
export ANALYTICS_OUTCOME_EVAL_ENABLED=true
export POSITION_MGMT_MODE=shadow
exec python -u bot.py

# TradingView -> BCS bot (adaptive)

This project includes:
- Python virtual environment: `.venv`
- A starter bot script: `bot.py`
- Dependency file: `requirements.txt`
- Environment template: `.env.example`

## What this bot does

1. Loads stocks universe from BCS source (`BCS_STOCKS_API_URL`) or automatically from MOEX board `TQBR` (all main Russian stocks), with local cache fallback.
2. Filters risky symbols (deterministic status filter + optional AI risk filter via OpenAI API).
3. Pulls trend indicators from TradingView (`EMA20/EMA50`, `MACD`, `RSI`, `ADX`, price).
4. Scores symbols and builds signals:
   - `BUY` when score >= strategy buy threshold
   - `SELL` when score <= strategy sell threshold
   - `HOLD` otherwise
5. Adapts strategy parameters over time via OpenAI API (`OPENAI_REBALANCE_CYCLES`).
6. Executes in selected mode:
   - `TRADING_MODE=paper`: virtual balance + virtual positions in `paper_state.json`
   - `TRADING_MODE=bcs`: sends market orders to BCS API
7. Sends signals to Telegram via `aiogram`.

## Quick start

```bash
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
python bot.py
```

## Telegram setup

1. Create bot via [@BotFather](https://t.me/BotFather) and get token.
2. Add bot to your chat (or use direct message with bot).
3. Set in `.env`:
   - `TELEGRAM_BOT_TOKEN=...`
   - `TELEGRAM_CHAT_ID=...`
   - `TELEGRAM_NOTIFY_HOLD=false` (set `true` if you also want HOLD signals)

## OpenAI setup

Set in `.env`:
- `OPENAI_API_KEY=...`
- `OPENAI_MODEL=gpt-4o-mini`
- `ENABLE_AI_RISK_FILTER=true`
- `OPENAI_REBALANCE_CYCLES=20`

## BCS universe setup

- Set `BCS_STOCKS_API_URL` to endpoint returning list of stocks (JSON).
- If unavailable, bot uses cache (`BCS_STOCKS_CACHE_FILE`) or fallback symbols from `TV_SYMBOLS`.

## Important before real trading

- Use `TRADING_MODE=paper` to test safely with no real orders.
- Keep real secrets only in `.env`, not in `.env.example`.
- Keep `BCS_DRY_RUN=true` even in `bcs` mode until endpoint/format are confirmed.
- Replace `BCS_API_BASE_URL`, `BCS_API_TOKEN`, `BCS_ACCOUNT_ID` in `.env`.
- Verify exact BCS order endpoint and payload schema for your account.
- Add risk controls (max daily loss, max open positions, stop-loss, take-profit).

## Safety note

This is an educational starter, not financial advice.

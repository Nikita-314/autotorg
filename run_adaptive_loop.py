#!/usr/bin/env python3
"""
Фоновый adaptive loop: периодически вызывает AdaptiveEngine.step().
Запуск из корня проекта:  python run_adaptive_loop.py

Переменные: см. adaptive/config.py (ADAPTIVE_MODE, ADAPTIVE_LOOP_INTERVAL_SEC, …).
"""
from __future__ import annotations

import asyncio
import logging
import os
import sys

from dotenv import load_dotenv

# корень репозитория в PYTHONPATH
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from adaptive.config import AdaptiveConfig  # noqa: E402
from adaptive.engine import AdaptiveEngine  # noqa: E402


async def _maybe_telegram(msg: str) -> None:
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat = os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat or not msg:
        return
    try:
        from aiogram import Bot

        bot = Bot(token=token)
        await bot.send_message(chat_id=int(chat), text=msg[:3500])
        await bot.session.close()
    except Exception as exc:
        logging.getLogger("adaptive-loop").warning("telegram notify failed: %s", exc)


async def main() -> None:
    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = AdaptiveConfig.from_env()
    eng = AdaptiveEngine(cfg)
    log = logging.getLogger("adaptive-loop")
    if cfg.adaptive_mode == "off":
        log.warning("ADAPTIVE_MODE=off, exiting")
        return
    while True:
        try:
            out = eng.step()
            log.info("adaptive step: %s", out.get("status"))
            hint = out.get("telegram_hint")
            if hint:
                await _maybe_telegram(hint)
        except Exception as exc:
            log.exception("adaptive step failed: %s", exc)
        await asyncio.sleep(max(30, cfg.loop_interval_sec))


if __name__ == "__main__":
    asyncio.run(main())

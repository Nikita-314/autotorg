#!/usr/bin/env python3
"""
Минимальный Telegram sidecar: /balance, /adaptive_status, /adaptive_reset.
Запуск: python telegram_adaptive_runner.py

Нужны TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, .env с путями к analytics.db и paper balance.
"""
from __future__ import annotations

import asyncio
import logging
import os
import sys

from dotenv import load_dotenv

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from aiogram import Bot, Dispatcher, Router  # noqa: E402
from aiogram.filters import BaseFilter, Command  # noqa: E402
from aiogram.types import Message  # noqa: E402

from adaptive.engine import AdaptiveEngine  # noqa: E402


class AuthorizedChat(BaseFilter):
    def __init__(self) -> None:
        self._expected = os.getenv("TELEGRAM_CHAT_ID", "").strip()

    async def __call__(self, message: Message) -> bool:
        if not self._expected:
            return True
        return str(message.chat.id) == self._expected


def _engine() -> AdaptiveEngine:
    return AdaptiveEngine()


def _split_telegram(text: str, limit: int = 3500) -> list[str]:
    lines = text.split("\n")
    chunks: list[str] = []
    buf: list[str] = []
    size = 0
    for ln in lines:
        if size + len(ln) + 1 > limit and buf:
            chunks.append("\n".join(buf))
            buf = [ln]
            size = len(ln)
        else:
            buf.append(ln)
            size += len(ln) + 1
    if buf:
        chunks.append("\n".join(buf))
    return chunks or [""]


async def cmd_balance(message: Message) -> None:
    await message.answer(_engine().balance_status_text())


async def cmd_adaptive_status(message: Message) -> None:
    for part in _split_telegram(_engine().adaptive_status_text()):
        await message.answer(part)


async def cmd_adaptive_reset(message: Message) -> None:
    await message.answer(_engine().adaptive_reset())


async def main() -> None:
    load_dotenv()
    logging.basicConfig(level=logging.INFO)
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        raise SystemExit("TELEGRAM_BOT_TOKEN missing")
    bot = Bot(token=token)
    dp = Dispatcher()
    auth = AuthorizedChat()
    r = Router()
    r.message.register(cmd_balance, Command("balance"), auth)
    r.message.register(cmd_adaptive_status, Command("adaptive_status"), auth)
    r.message.register(cmd_adaptive_reset, Command("adaptive_reset"), auth)
    dp.include_router(r)
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())

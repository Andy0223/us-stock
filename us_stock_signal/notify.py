from __future__ import annotations

import os
import re
from typing import Any

import requests


def send_telegram_message(config: dict[str, Any], text: str) -> list[dict[str, Any]]:
    token = os.getenv(str(config.get("bot_token_env", "US_STOCK_TELEGRAM_BOT_TOKEN")), "")
    chat_id = os.getenv(str(config.get("chat_id_env", "US_STOCK_TELEGRAM_CHAT_ID")), "")
    if not token or not chat_id:
        raise RuntimeError("Telegram token or chat id is missing from environment.")
    if not re.fullmatch(r"\d+:[A-Za-z0-9_-]+", token):
        raise RuntimeError("Telegram bot token format is invalid; expected numeric bot id, colon, and token secret.")

    responses = []
    for chunk in split_telegram_text(text):
        try:
            response = requests.post(
                f"https://api.telegram.org/bot{token}/sendMessage",
                json={
                    "chat_id": chat_id,
                    "text": chunk,
                    "disable_web_page_preview": True,
                },
                timeout=15,
            )
            response.raise_for_status()
        except requests.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else "unknown"
            raise RuntimeError(f"Telegram send failed with HTTP {status}; check bot token and chat id.") from exc
        responses.append(response.json())
    return responses


def split_telegram_text(text: str, limit: int = 3800) -> list[str]:
    if len(text) <= limit:
        return [text]
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0
    for line in text.splitlines():
        candidate_len = len(line) + 1
        if current and current_len + candidate_len > limit:
            chunks.append("\n".join(current))
            current = []
            current_len = 0
        if candidate_len > limit:
            for start in range(0, len(line), limit):
                chunks.append(line[start : start + limit])
            continue
        current.append(line)
        current_len += candidate_len
    if current:
        chunks.append("\n".join(current))
    return chunks

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Iterable


logger = logging.getLogger(__name__)


def collect_symbol_news(
    symbols: Iterable[str],
    max_symbols: int = 14,
    max_items_per_symbol: int = 2,
) -> dict[str, Any]:
    ticker_list = unique_symbols(symbols)[: max(0, int(max_symbols))]
    if not ticker_list:
        return {
            "items": [],
            "source_status": {"symbols_requested": 0, "symbols_returned": 0, "errors": []},
            "note": "No symbols selected for news review.",
        }

    try:
        import yfinance as yf
    except Exception as exc:  # pragma: no cover - optional dependency issue
        return {
            "items": [],
            "source_status": {"symbols_requested": len(ticker_list), "symbols_returned": 0, "errors": [str(exc)]},
            "note": "yfinance is unavailable; news review skipped.",
        }

    items: list[dict[str, Any]] = []
    errors: list[str] = []
    returned_symbols: set[str] = set()
    for ticker in ticker_list:
        try:
            raw_items = getattr(yf.Ticker(ticker), "news", []) or []
        except Exception as exc:  # pragma: no cover - network/provider dependent
            logger.debug("News fetch failed for %s: %s", ticker, exc)
            errors.append(f"{ticker}: {exc}")
            continue

        normalized = [item for item in (normalize_news_item(ticker, raw) for raw in raw_items) if item]
        if normalized:
            returned_symbols.add(ticker)
            items.extend(normalized[: max(1, int(max_items_per_symbol))])

    return {
        "items": dedupe_news(items),
        "source_status": {
            "symbols_requested": len(ticker_list),
            "symbols_returned": len(returned_symbols),
            "errors": errors[:8],
        },
        "note": "News is sourced from yfinance ticker news when available; it is a reminder feed, not complete market news coverage.",
    }


def normalize_news_item(ticker: str, item: Any) -> dict[str, Any] | None:
    if not isinstance(item, dict):
        return None
    content = item.get("content") if isinstance(item.get("content"), dict) else item
    title = first_text(content.get("title"), item.get("title"))
    if not title:
        return None
    publisher = first_text(
        nested_text(content.get("provider"), "displayName"),
        content.get("publisher"),
        item.get("publisher"),
    )
    link = first_text(
        nested_text(content.get("canonicalUrl"), "url"),
        nested_text(content.get("clickThroughUrl"), "url"),
        content.get("link"),
        item.get("link"),
    )
    published_at = normalize_published_time(
        content.get("pubDate")
        or content.get("displayTime")
        or item.get("providerPublishTime")
        or item.get("pubDate")
    )
    summary = first_text(content.get("summary"), item.get("summary"))
    return {
        "ticker": ticker,
        "title": title.strip(),
        "publisher": publisher.strip() if publisher else "",
        "published_at": published_at,
        "link": link.strip() if link else "",
        "summary": summary.strip() if summary else "",
    }


def normalize_published_time(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (int, float)):
        try:
            return datetime.fromtimestamp(float(value), tz=timezone.utc).isoformat(timespec="seconds")
        except Exception:
            return ""
    text = str(value).strip()
    if not text:
        return ""
    return text


def dedupe_news(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for item in items:
        key = (str(item.get("ticker", "")), str(item.get("title", "")).lower())
        if key in seen:
            continue
        seen.add(key)
        result.append(item)
    return result


def first_text(*values: Any) -> str:
    for value in values:
        if isinstance(value, str) and value.strip():
            return value
    return ""


def nested_text(value: Any, key: str) -> str:
    if isinstance(value, dict):
        raw = value.get(key)
        if isinstance(raw, str):
            return raw
    return ""


def unique_symbols(symbols: Iterable[str]) -> list[str]:
    result: list[str] = []
    for symbol in symbols:
        ticker = str(symbol).strip().upper()
        if ticker and ticker not in result:
            result.append(ticker)
    return result

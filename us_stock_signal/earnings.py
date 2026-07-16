from __future__ import annotations

import re
from datetime import date, timedelta
from typing import Any
from urllib.request import Request, urlopen

import pandas as pd


DEFAULT_URL = "https://www.alphalab.site/earnings"


def collect_earnings_calendar(
    symbols: list[str],
    as_of: date,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    cfg = config or {}
    if not bool(cfg.get("enabled", True)):
        return empty_calendar(enabled=False, note="earnings calendar disabled")

    url = str(cfg.get("url", DEFAULT_URL))
    timeout = float(cfg.get("timeout_seconds", 20))
    lookahead_days = int(cfg.get("lookahead_days", 14))
    max_rows = int(cfg.get("max_rows", 80))
    watch_symbols = normalize_symbols(symbols)

    try:
        html = fetch_page(url, timeout)
        rows = parse_alphalab_earnings(html)
    except Exception as exc:  # pragma: no cover - network/page dependent
        result = empty_calendar(enabled=True, note=f"fetch_or_parse_error: {exc}")
        result["source_status"]["error"] = str(exc)
        return result

    start = as_of
    end = as_of + timedelta(days=max(0, lookahead_days))
    upcoming = [row for row in rows if start <= row["earnings_date"] <= end]
    today = [row for row in upcoming if row["earnings_date"] == as_of]
    tomorrow = [row for row in upcoming if row["earnings_date"] == as_of + timedelta(days=1)]
    future_before = [row for row in upcoming if row["earnings_date"] > as_of and row["release_time"] == "before_open"]
    next_before_open = rows_on_earliest_date(future_before)
    watched = [row for row in upcoming if row["ticker"] in watch_symbols]
    high_attention = [row for row in upcoming if row["ticker"] in high_attention_symbols()]

    return {
        "enabled": True,
        "source": "AlphaLab earnings calendar",
        "source_url": url,
        "source_status": {
            "rows_parsed": len(rows),
            "upcoming_rows": len(upcoming),
            "watch_symbols": len(watch_symbols),
            "watched_rows": len(watched),
        },
        "as_of": as_of.isoformat(),
        "lookahead_days": lookahead_days,
        "today": serialize_rows(today[:max_rows]),
        "tomorrow": serialize_rows(tomorrow[:max_rows]),
        "next_before_open": serialize_rows(next_before_open[:max_rows]),
        "upcoming": serialize_rows(upcoming[:max_rows]),
        "watched": serialize_rows(watched[:max_rows]),
        "high_attention": serialize_rows(high_attention[:max_rows]),
        "risk_note": "財報前後波動放大；候選股若即將財報，預設不追高，除非盤後確認突破且風控允許。",
    }


def fetch_page(url: str, timeout: float) -> str:
    request = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(request, timeout=timeout) as response:
        return response.read().decode("utf-8", errors="replace")


def parse_alphalab_earnings(html: str) -> list[dict[str, Any]]:
    decoded = html.encode("utf-8").decode("unicode_escape", errors="ignore")
    rows: list[dict[str, Any]] = []
    for date_text, array_text in re.findall(
        r"\"(20\d{2}-\d{2}-\d{2})\":\[(.*?)\](?=,\"20|\},\"total|\}\])",
        decoded,
        flags=re.S,
    ):
        try:
            earnings_date = date.fromisoformat(date_text)
        except ValueError:
            continue
        for ticker, company, release_time in re.findall(
            r"\"ticker\":\"([^\"]+)\".*?\"company_name\":\"([^\"]+)\".*?\"earnings_release_time\":\"([^\"]+)\"",
            array_text,
            flags=re.S,
        ):
            normalized = normalize_symbol(ticker)
            if not normalized:
                continue
            rows.append(
                {
                    "earnings_date": earnings_date,
                    "date": earnings_date.isoformat(),
                    "ticker": normalized,
                    "raw_ticker": ticker,
                    "company": company,
                    "release_time": release_time,
                    "release_time_label": release_label(release_time),
                }
            )
    rows = dedupe_rows(rows)
    return sorted(rows, key=lambda item: (item["earnings_date"], item["release_time"], item["ticker"]))


def dedupe_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, str, str]] = set()
    result = []
    for row in rows:
        key = (str(row.get("date", "")), str(row.get("ticker", "")), str(row.get("release_time", "")))
        if key in seen:
            continue
        seen.add(key)
        result.append(row)
    return result


def serialize_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for row in rows:
        item = dict(row)
        if isinstance(item.get("earnings_date"), date):
            item["earnings_date"] = item["earnings_date"].isoformat()
        result.append(item)
    return result


def normalize_symbols(symbols: list[str]) -> set[str]:
    return {normalize_symbol(symbol) for symbol in symbols if normalize_symbol(symbol)}


def normalize_symbol(symbol: Any) -> str:
    text = str(symbol or "").strip().upper()
    if not text or text in {"CASH", "MARGIN_BALANCE"}:
        return ""
    return text.replace(".", "-")


def release_label(value: str) -> str:
    labels = {
        "before_open": "盤前",
        "after_close": "盤後",
    }
    return labels.get(str(value), str(value))


def high_attention_symbols() -> set[str]:
    return {
        "AAPL",
        "ABBV",
        "AMD",
        "AMZN",
        "ASML",
        "AVGO",
        "CRM",
        "GOOGL",
        "INTC",
        "LLY",
        "META",
        "MRVL",
        "MSFT",
        "NFLX",
        "NVDA",
        "PLTR",
        "QCOM",
        "SMCI",
        "TSLA",
        "TSM",
        "UNH",
    }


def empty_calendar(enabled: bool, note: str) -> dict[str, Any]:
    return {
        "enabled": enabled,
        "source": "AlphaLab earnings calendar",
        "source_url": DEFAULT_URL,
        "source_status": {"rows_parsed": 0, "upcoming_rows": 0, "watch_symbols": 0, "watched_rows": 0, "note": note},
        "today": [],
        "tomorrow": [],
        "next_before_open": [],
        "upcoming": [],
        "watched": [],
        "high_attention": [],
        "risk_note": note,
    }


def rows_to_frame(calendar: dict[str, Any]) -> pd.DataFrame:
    rows = calendar.get("upcoming", []) if isinstance(calendar, dict) else []
    if not rows:
        return pd.DataFrame(columns=["date", "ticker", "company", "release_time", "release_time_label"])
    return pd.DataFrame(rows)


def rows_on_earliest_date(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        return []
    first_date = min(row["earnings_date"] for row in rows)
    return [row for row in rows if row["earnings_date"] == first_date]

from __future__ import annotations

import math
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pandas as pd


WATCHLIST_COLUMNS = [
    "created_at",
    "valid_for",
    "ticker",
    "name",
    "action",
    "priority",
    "reason",
    "trigger_low",
    "trigger_high",
    "stop_loss",
    "no_chase_above",
    "source_section",
    "status",
]


def load_active_watchlist(path: str | Path, as_of: date) -> list[dict[str, Any]]:
    watchlist_path = Path(path).expanduser()
    if not watchlist_path.exists():
        return []
    try:
        frame = pd.read_csv(watchlist_path)
    except Exception:
        return []
    if frame.empty:
        return []
    for column in WATCHLIST_COLUMNS:
        if column not in frame:
            frame[column] = ""
    frame["ticker"] = frame["ticker"].fillna("").astype(str).str.strip().str.upper()
    frame["valid_for"] = frame["valid_for"].fillna("").astype(str)
    frame = frame[frame["ticker"].ne("")]
    active = frame[frame["valid_for"].eq(as_of.isoformat())].copy()
    if active.empty:
        return []
    return records(active)


def build_next_day_watchlist(context: dict[str, Any], valid_for: date) -> list[dict[str, Any]]:
    review = context.get("after_close_review", {}) if isinstance(context.get("after_close_review"), dict) else {}
    rows: list[dict[str, Any]] = []

    for row in review.get("missed_sell_candidates", []) or []:
        rows.append(watchlist_row(context, valid_for, row, action="sell_or_reduce", priority=1, source="missed_sell"))
    for row in review.get("missed_add_candidates", []) or []:
        rows.append(watchlist_row(context, valid_for, row, action="add_watch", priority=2, source="missed_add"))
    for row in review.get("missed_buy_candidates", []) or []:
        rows.append(watchlist_row(context, valid_for, row, action="buy_watch", priority=3, source="missed_buy"))
    for row in review.get("big_losers_1d", []) or []:
        rows.append(watchlist_row(context, valid_for, row, action="risk_news_watch", priority=4, source="big_loser"))
    for row in review.get("big_gainers_1d", []) or []:
        rows.append(watchlist_row(context, valid_for, row, action="momentum_news_watch", priority=5, source="big_gainer"))

    return dedupe_watchlist(rows)[: int(context.get("watchlist_max_rows", 30) or 30)]


def evaluate_watchlist(
    active_rows: list[dict[str, Any]],
    context: dict[str, Any],
    price_map: dict[str, float] | None = None,
) -> dict[str, Any]:
    if not active_rows:
        return {
            "items": [],
            "triggered": [],
            "missed": [],
            "still_waiting": [],
            "no_price": [],
            "summary": {"active_count": 0},
        }

    merged_price_map = latest_price_map(context)
    merged_price_map.update(price_map or {})
    evaluated: list[dict[str, Any]] = []
    for row in active_rows:
        item = dict(row)
        ticker = str(item.get("ticker", "")).upper()
        close = merged_price_map.get(ticker, math.nan)
        action = str(item.get("action", ""))
        trigger_low = safe_float(item.get("trigger_low"))
        trigger_high = safe_float(item.get("trigger_high"))
        stop_loss = safe_float(item.get("stop_loss"))
        no_chase = safe_float(item.get("no_chase_above"))

        status = "waiting"
        note = "尚未觸發。"
        if is_finite(close):
            if action in {"sell_or_reduce", "risk_news_watch"} and is_finite(stop_loss) and close <= stop_loss:
                status = "triggered"
                note = "價格跌破停損/風險線，優先處理。"
            elif action in {"buy_watch", "add_watch"} and in_range(close, trigger_low, trigger_high):
                status = "triggered"
                note = "價格進入觸發區。"
            elif action in {"buy_watch", "add_watch"} and is_finite(no_chase) and close > no_chase:
                status = "missed"
                note = "已高於不追價，視為錯過，等回測。"
            elif action in {"momentum_news_watch"}:
                status = "waiting"
                note = "只做消息/延續性追蹤，不直接追價。"
            elif action in {"factor_watch"}:
                status = "waiting"
                note = "閉環早期因子追蹤，需等價格觸發或盤後確認。"
        else:
            status = "no_price"
            note = "沒有價格資料。"

        item["current_price"] = close
        item["evaluation_status"] = status
        item["evaluation_note"] = note
        evaluated.append(item)

    triggered = [row for row in evaluated if row.get("evaluation_status") == "triggered"]
    missed = [row for row in evaluated if row.get("evaluation_status") == "missed"]
    waiting = [row for row in evaluated if row.get("evaluation_status") == "waiting"]
    no_price = [row for row in evaluated if row.get("evaluation_status") == "no_price"]
    return {
        "items": evaluated,
        "triggered": triggered,
        "missed": missed,
        "still_waiting": waiting,
        "no_price": no_price,
        "summary": {
            "active_count": len(evaluated),
            "triggered_count": len(triggered),
            "missed_count": len(missed),
            "waiting_count": len(waiting),
            "no_price_count": len(no_price),
        },
    }


def write_watchlist(path: str | Path, rows: list[dict[str, Any]]) -> Path:
    watchlist_path = Path(path).expanduser()
    watchlist_path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    if frame.empty:
        frame = pd.DataFrame(columns=WATCHLIST_COLUMNS)
    for column in WATCHLIST_COLUMNS:
        if column not in frame:
            frame[column] = ""
    frame = frame.loc[:, WATCHLIST_COLUMNS]
    frame.to_csv(watchlist_path, index=False)
    return watchlist_path


def next_calendar_day(as_of: date) -> date:
    return as_of + timedelta(days=1)


def watchlist_row(
    context: dict[str, Any],
    valid_for: date,
    row: dict[str, Any],
    action: str,
    priority: int,
    source: str,
) -> dict[str, Any]:
    return {
        "created_at": str(context.get("as_of", "")),
        "valid_for": valid_for.isoformat(),
        "ticker": str(row.get("ticker", "")).upper(),
        "name": row.get("name", ""),
        "action": action,
        "priority": priority,
        "reason": reason_for(action, row),
        "trigger_low": first_present(row, ["entry_low", "close"]),
        "trigger_high": first_present(row, ["entry_high", "close"]),
        "stop_loss": row.get("stop_loss", ""),
        "no_chase_above": row.get("no_chase_above", ""),
        "source_section": source,
        "status": "active",
    }


def reason_for(action: str, row: dict[str, Any]) -> str:
    if action == "sell_or_reduce":
        return str(row.get("reason") or "盤後列為該賣/該減碼。")
    if action == "add_watch":
        return str(row.get("reason") or "盤後列為可加碼追蹤。")
    if action == "buy_watch":
        return str(row.get("reason") or "盤後列為該買未買追蹤。")
    if action == "risk_news_watch":
        return "單日大跌，隔日追蹤是否有消息或風險延續。"
    if action == "factor_watch":
        return str(row.get("reason") or "閉環早期因子雷達追蹤。")
    return "單日大漲，隔日追蹤是否有消息或延續性。"


def dedupe_watchlist(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    best: dict[str, dict[str, Any]] = {}
    for row in rows:
        ticker = str(row.get("ticker", "")).strip().upper()
        if not ticker:
            continue
        previous = best.get(ticker)
        if previous is None or int(row.get("priority", 99)) < int(previous.get("priority", 99)):
            best[ticker] = row
    return sorted(best.values(), key=lambda item: (int(item.get("priority", 99)), str(item.get("ticker", ""))))


def latest_price_map(context: dict[str, Any]) -> dict[str, float]:
    result: dict[str, float] = {}
    for section, symbol_key in [("holdings_review", "ticker"), ("top_candidates", "ticker")]:
        for row in context.get(section, []) or []:
            ticker = str(row.get(symbol_key, "")).strip().upper()
            if not ticker:
                continue
            close = safe_float(row.get("close"))
            if is_finite(close):
                result[ticker] = close
    return result


def first_present(row: dict[str, Any], keys: list[str]) -> Any:
    for key in keys:
        value = row.get(key)
        if is_finite(safe_float(value)):
            return value
    return ""


def in_range(value: float, low: float, high: float) -> bool:
    if not is_finite(value) or not is_finite(low) or not is_finite(high):
        return False
    lower = min(low, high)
    upper = max(low, high)
    return lower <= value <= upper


def safe_float(value: Any, default: float = math.nan) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if math.isfinite(parsed) else default


def is_finite(value: Any) -> bool:
    return math.isfinite(safe_float(value))


def records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    cleaned = frame.copy()
    cleaned = cleaned.where(pd.notna(cleaned), None)
    return cleaned.to_dict("records")

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd


TRADE_COLUMNS = [
    "date",
    "side",
    "ticker",
    "name",
    "shares",
    "price",
    "amount",
    "account_type",
    "source",
]


def load_trade_log(path: str | Path, as_of: date) -> dict[str, Any]:
    trade_path = Path(path).expanduser()
    if not trade_path.exists():
        return {
            "as_of": as_of.isoformat(),
            "path": str(trade_path),
            "rows": [],
            "buy_value": 0.0,
            "sell_value": 0.0,
            "net_cash_flow": 0.0,
            "symbols_bought": [],
            "symbols_sold": [],
            "round_trip_symbols": [],
            "source_status": {"status": "missing", "rows_total": 0, "rows_for_date": 0},
        }

    raw = pd.read_csv(trade_path)
    frame = normalize_trade_frame(raw)
    day_frame = frame[frame["date"].eq(as_of.isoformat())].copy() if not frame.empty else frame
    buy_rows = day_frame[day_frame["side"].eq("buy")] if not day_frame.empty else day_frame
    sell_rows = day_frame[day_frame["side"].eq("sell")] if not day_frame.empty else day_frame
    buy_value = float(buy_rows["amount"].abs().sum()) if not buy_rows.empty else 0.0
    sell_value = float(sell_rows["amount"].abs().sum()) if not sell_rows.empty else 0.0
    bought = sorted(set(buy_rows["ticker"].dropna().astype(str))) if not buy_rows.empty else []
    sold = sorted(set(sell_rows["ticker"].dropna().astype(str))) if not sell_rows.empty else []
    return {
        "as_of": as_of.isoformat(),
        "path": str(trade_path),
        "rows": records(day_frame),
        "buy_value": buy_value,
        "sell_value": sell_value,
        "net_cash_flow": sell_value - buy_value,
        "symbols_bought": bought,
        "symbols_sold": sold,
        "round_trip_symbols": sorted(set(bought) & set(sold)),
        "source_status": {
            "status": "ok",
            "rows_total": int(len(frame)),
            "rows_for_date": int(len(day_frame)),
        },
    }


def normalize_trade_frame(raw: pd.DataFrame) -> pd.DataFrame:
    frame = raw.copy()
    for column in TRADE_COLUMNS:
        if column not in frame:
            frame[column] = "" if column in {"date", "side", "ticker", "name", "account_type", "source"} else 0.0

    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.date.astype("string")
    frame["side"] = frame["side"].map(normalize_side)
    frame["ticker"] = frame["ticker"].fillna("").astype(str).str.strip().str.upper()
    frame["name"] = frame["name"].fillna("").astype(str).str.strip()
    frame["shares"] = pd.to_numeric(frame["shares"], errors="coerce").fillna(0.0)
    frame["price"] = pd.to_numeric(frame["price"], errors="coerce").fillna(0.0)
    frame["amount"] = pd.to_numeric(frame["amount"], errors="coerce")
    missing_amount = frame["amount"].isna()
    if missing_amount.any():
        gross = frame["shares"].abs() * frame["price"].abs()
        sign = frame["side"].map({"buy": -1.0, "sell": 1.0}).fillna(0.0)
        frame.loc[missing_amount, "amount"] = gross[missing_amount] * sign[missing_amount]
    frame["amount"] = frame["amount"].fillna(0.0)
    frame["account_type"] = frame["account_type"].fillna("").astype(str).str.strip()
    frame["source"] = frame["source"].fillna("").astype(str).str.strip()
    frame = frame[frame["ticker"].ne("") & frame["side"].isin(["buy", "sell"])].copy()
    return frame[TRADE_COLUMNS].reset_index(drop=True)


def normalize_side(value: Any) -> str:
    text = str(value).strip().lower()
    if text in {"buy", "b", "買", "買進", "買入"}:
        return "buy"
    if text in {"sell", "s", "賣", "賣出"}:
        return "sell"
    return text


def records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    clean = frame.copy()
    clean = clean.where(pd.notna(clean), None)
    return clean.to_dict("records")

from __future__ import annotations

import logging
import math
import time
from datetime import date
from typing import Any, Iterable

import pandas as pd


logger = logging.getLogger(__name__)


def collect_options_review(
    symbols: Iterable[str],
    price_lookup: dict[str, float] | None = None,
    as_of: date | None = None,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    cfg = config or {}
    if not bool(cfg.get("enabled", True)):
        return empty_options_review(enabled=False, note="Options review disabled.")

    try:
        import yfinance as yf
    except Exception as exc:  # pragma: no cover - local dependency dependent
        review = empty_options_review(enabled=True, note=f"yfinance unavailable: {exc}")
        review["source_status"]["error"] = str(exc)
        return review

    price_map = {str(k).upper(): safe_float(v) for k, v in (price_lookup or {}).items()}
    symbol_list = unique_symbols(symbols)[: max(1, int(cfg.get("max_symbols", 14)))]
    pause_seconds = float(cfg.get("pause_seconds", 0.05))
    rows: list[dict[str, Any]] = []
    errors: dict[str, str] = {}
    unavailable: list[str] = []

    for symbol in symbol_list:
        try:
            row = analyze_symbol_options(yf, symbol, price_map.get(symbol, math.nan), as_of, cfg)
        except Exception as exc:  # pragma: no cover - provider/network dependent
            logger.debug("Options review failed for %s", symbol, exc_info=True)
            errors[symbol] = str(exc)
            continue

        if row.get("data_available"):
            rows.append(row)
        else:
            unavailable.append(symbol)
            if row.get("note"):
                errors[symbol] = str(row.get("note"))

        if pause_seconds > 0:
            time.sleep(pause_seconds)

    rows = sorted(
        rows,
        key=lambda item: (
            abs(safe_float(item.get("signal_score"), 0.0)),
            safe_float(item.get("total_option_volume"), 0.0),
        ),
        reverse=True,
    )
    max_alerts = int(cfg.get("max_alerts", 8))
    bullish = [row for row in rows if row.get("direction_signal") == "bullish"][:max_alerts]
    bearish = [row for row in rows if row.get("direction_signal") == "bearish"][:max_alerts]
    high_iv = [row for row in rows if bool(row.get("high_iv"))][:max_alerts]

    return {
        "enabled": True,
        "source": "yfinance option chains",
        "scope_note": "期權檢查只涵蓋目前持股、前排候選與盤後追蹤標的；不是全市場 options flow。",
        "source_status": {
            "symbols_requested": len(symbol_list),
            "symbols_returned": len(rows),
            "symbols_without_options_or_data": len(unavailable),
            "errors": errors,
        },
        "summary": {
            "bullish_count": len(bullish),
            "bearish_count": len(bearish),
            "high_iv_count": len(high_iv),
        },
        "rows": rows,
        "bullish_confirmations": bullish,
        "bearish_warnings": bearish,
        "high_iv_watchlist": high_iv,
    }


def empty_options_review(enabled: bool, note: str) -> dict[str, Any]:
    return {
        "enabled": enabled,
        "source": "yfinance option chains",
        "scope_note": "期權檢查只涵蓋目前持股、前排候選與盤後追蹤標的；不是全市場 options flow。",
        "source_status": {
            "symbols_requested": 0,
            "symbols_returned": 0,
            "symbols_without_options_or_data": 0,
            "errors": {},
            "note": note,
        },
        "summary": {"bullish_count": 0, "bearish_count": 0, "high_iv_count": 0},
        "rows": [],
        "bullish_confirmations": [],
        "bearish_warnings": [],
        "high_iv_watchlist": [],
    }


def analyze_symbol_options(
    yf: Any,
    symbol: str,
    price: float,
    as_of: date | None,
    cfg: dict[str, Any],
) -> dict[str, Any]:
    ticker = yf.Ticker(symbol)
    expirations = choose_expirations(
        list(getattr(ticker, "options", []) or []),
        as_of=as_of,
        min_days=int(cfg.get("min_days_to_expiry", 3)),
        max_days=int(cfg.get("max_days_to_expiry", 45)),
        max_expirations=int(cfg.get("max_expirations", 3)),
    )
    if not expirations:
        return {"ticker": symbol, "data_available": False, "note": "no option expirations"}

    frames: list[pd.DataFrame] = []
    for expiry in expirations:
        try:
            chain = ticker.option_chain(expiry)
        except Exception as exc:  # pragma: no cover - provider/network dependent
            logger.debug("Option chain failed for %s %s: %s", symbol, expiry, exc)
            continue
        calls = normalize_chain_frame(getattr(chain, "calls", pd.DataFrame()), "call", expiry, price, as_of)
        puts = normalize_chain_frame(getattr(chain, "puts", pd.DataFrame()), "put", expiry, price, as_of)
        if not calls.empty:
            frames.append(calls)
        if not puts.empty:
            frames.append(puts)

    if not frames:
        return {"ticker": symbol, "data_available": False, "note": "no option chain rows"}

    data = pd.concat(frames, ignore_index=True)
    row = summarize_options(symbol, data, price, expirations, cfg)
    row["data_available"] = True
    return row


def choose_expirations(
    expirations: list[str],
    as_of: date | None,
    min_days: int,
    max_days: int,
    max_expirations: int,
) -> list[str]:
    today = as_of or date.today()
    dated: list[tuple[str, int]] = []
    for expiry in expirations:
        try:
            expiry_date = date.fromisoformat(str(expiry))
        except ValueError:
            continue
        dated.append((str(expiry), (expiry_date - today).days))

    in_window = [expiry for expiry, dte in dated if min_days <= dte <= max_days]
    if in_window:
        return in_window[: max(1, max_expirations)]
    future = [expiry for expiry, dte in dated if dte >= 0]
    return future[: max(1, max_expirations)]


def normalize_chain_frame(frame: pd.DataFrame, side: str, expiry: str, price: float, as_of: date | None) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame()
    result = frame.copy()
    result["side"] = side
    result["expiry"] = expiry
    try:
        expiry_date = date.fromisoformat(str(expiry))
        result["dte"] = (expiry_date - (as_of or date.today())).days
    except ValueError:
        result["dte"] = math.nan
    for column in ["strike", "volume", "openInterest", "impliedVolatility", "bid", "ask", "lastPrice"]:
        if column not in result:
            result[column] = 0.0
        result[column] = pd.to_numeric(result[column], errors="coerce").fillna(0.0)
    if is_finite(price) and price > 0:
        result["moneyness_abs"] = (result["strike"] / price - 1.0).abs()
    else:
        result["moneyness_abs"] = math.nan
    return result


def summarize_options(
    symbol: str,
    data: pd.DataFrame,
    price: float,
    expirations: list[str],
    cfg: dict[str, Any],
) -> dict[str, Any]:
    near_atm_pct = float(cfg.get("near_atm_pct", 0.05))
    min_total_volume = float(cfg.get("min_total_contract_volume", 500))
    min_near_atm_volume = float(cfg.get("min_near_atm_volume", 100))
    bullish_ratio = float(cfg.get("bullish_call_put_ratio", 1.5))
    bearish_ratio = float(cfg.get("bearish_put_call_ratio", 1.3))
    high_iv_threshold = float(cfg.get("high_iv_threshold", 0.65))

    calls = data[data["side"].eq("call")].copy()
    puts = data[data["side"].eq("put")].copy()
    near = data[pd.to_numeric(data["moneyness_abs"], errors="coerce").le(near_atm_pct)].copy()
    near_calls = near[near["side"].eq("call")]
    near_puts = near[near["side"].eq("put")]

    call_volume = numeric_sum(calls, "volume")
    put_volume = numeric_sum(puts, "volume")
    call_oi = numeric_sum(calls, "openInterest")
    put_oi = numeric_sum(puts, "openInterest")
    near_call_volume = numeric_sum(near_calls, "volume")
    near_put_volume = numeric_sum(near_puts, "volume")
    total_volume = call_volume + put_volume
    total_oi = call_oi + put_oi

    near_iv_source = near[pd.to_numeric(near["impliedVolatility"], errors="coerce").gt(0)]
    all_iv_source = data[pd.to_numeric(data["impliedVolatility"], errors="coerce").gt(0)]
    avg_near_iv = numeric_mean(near_iv_source, "impliedVolatility")
    avg_iv = avg_near_iv if is_finite(avg_near_iv) else numeric_mean(all_iv_source, "impliedVolatility")
    high_iv = is_finite(avg_iv) and avg_iv >= high_iv_threshold

    strongest_call = strongest_contract(calls)
    strongest_put = strongest_contract(puts)
    strongest = choose_strongest_contract(strongest_call, strongest_put)

    call_put_ratio = safe_ratio(call_volume, put_volume)
    put_call_ratio = safe_ratio(put_volume, call_volume)
    near_call_put_ratio = safe_ratio(near_call_volume, near_put_volume)
    near_put_call_ratio = safe_ratio(near_put_volume, near_call_volume)
    call_put_oi_ratio = safe_ratio(call_oi, put_oi)

    bullish = (
        total_volume >= min_total_volume
        and near_call_volume >= min_near_atm_volume
        and call_put_ratio >= bullish_ratio
        and near_call_put_ratio >= 1.2
    )
    bearish = (
        total_volume >= min_total_volume
        and near_put_volume >= min_near_atm_volume
        and put_call_ratio >= bearish_ratio
        and near_put_call_ratio >= 1.2
    )

    if total_volume < min_total_volume:
        direction = "insufficient"
        alert = "期權量不足"
        score = 0.0
    elif bullish and bearish:
        direction = "mixed"
        alert = "多空期權同時放大"
        score = 0.0
    elif bullish:
        direction = "bullish"
        alert = "期權偏多確認"
        score = min(5.0, math.log1p(call_put_ratio) + math.log1p(max(near_call_put_ratio, 0.0)))
    elif bearish:
        direction = "bearish"
        alert = "期權偏空警訊"
        score = -min(5.0, math.log1p(put_call_ratio) + math.log1p(max(near_put_call_ratio, 0.0)))
    else:
        direction = "neutral"
        alert = "期權中性"
        score = 0.0

    if high_iv and direction in {"bullish", "neutral", "mixed"}:
        alert = f"{alert}，IV 偏高勿追"

    return {
        "ticker": symbol,
        "price_used": price if is_finite(price) else math.nan,
        "expirations_checked": len(expirations),
        "nearest_expiration": expirations[0] if expirations else "",
        "total_option_volume": int(total_volume),
        "total_open_interest": int(total_oi),
        "call_volume": int(call_volume),
        "put_volume": int(put_volume),
        "call_put_volume_ratio": round(call_put_ratio, 3),
        "put_call_volume_ratio": round(put_call_ratio, 3),
        "call_put_open_interest_ratio": round(call_put_oi_ratio, 3),
        "near_atm_call_volume": int(near_call_volume),
        "near_atm_put_volume": int(near_put_volume),
        "near_atm_call_put_ratio": round(near_call_put_ratio, 3),
        "near_atm_put_call_ratio": round(near_put_call_ratio, 3),
        "avg_near_atm_iv": round(avg_iv, 4) if is_finite(avg_iv) else math.nan,
        "high_iv": high_iv,
        "strongest_contract_side": strongest.get("side", ""),
        "strongest_contract_expiry": strongest.get("expiry", ""),
        "strongest_contract_strike": strongest.get("strike", math.nan),
        "strongest_contract_volume": int(safe_float(strongest.get("volume"), 0.0)),
        "strongest_contract_open_interest": int(safe_float(strongest.get("openInterest"), 0.0)),
        "strongest_contract_iv": round(safe_float(strongest.get("impliedVolatility"), math.nan), 4),
        "direction_signal": direction,
        "signal_score": round(score, 3),
        "alert": alert,
    }


def strongest_contract(frame: pd.DataFrame) -> dict[str, Any]:
    if frame is None or frame.empty:
        return {}
    output = frame.copy()
    output["volume"] = pd.to_numeric(output["volume"], errors="coerce").fillna(0.0)
    output["openInterest"] = pd.to_numeric(output["openInterest"], errors="coerce").fillna(0.0)
    output = output[output["volume"].gt(0)].copy()
    if output.empty:
        return {}
    output["volume_oi_ratio"] = output["volume"] / output["openInterest"].clip(lower=1)
    return output.sort_values(["volume", "volume_oi_ratio"], ascending=[False, False]).iloc[0].to_dict()


def choose_strongest_contract(call: dict[str, Any], put: dict[str, Any]) -> dict[str, Any]:
    if not call:
        return put or {}
    if not put:
        return call
    return call if safe_float(call.get("volume"), 0.0) >= safe_float(put.get("volume"), 0.0) else put


def unique_symbols(symbols: Iterable[str]) -> list[str]:
    result: list[str] = []
    for symbol in symbols:
        ticker = str(symbol).strip().upper()
        if not ticker or ticker in {"CASH", "MARGIN_BALANCE"}:
            continue
        if ticker.startswith("^") or "=" in ticker:
            continue
        if ticker not in result:
            result.append(ticker)
    return result


def numeric_sum(frame: pd.DataFrame, column: str) -> float:
    if frame is None or frame.empty or column not in frame:
        return 0.0
    value = pd.to_numeric(frame[column], errors="coerce").fillna(0.0).sum()
    return float(value) if math.isfinite(float(value)) else 0.0


def numeric_mean(frame: pd.DataFrame, column: str) -> float:
    if frame is None or frame.empty or column not in frame:
        return math.nan
    series = pd.to_numeric(frame[column], errors="coerce").dropna()
    series = series[series.gt(0)]
    if series.empty:
        return math.nan
    value = float(series.mean())
    return value if math.isfinite(value) else math.nan


def safe_ratio(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return float(numerator) if numerator > 0 else 0.0
    return float(numerator) / float(denominator)


def safe_float(value: Any, default: float = math.nan) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if math.isfinite(parsed) else default


def is_finite(value: Any) -> bool:
    return math.isfinite(safe_float(value))

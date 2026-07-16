from __future__ import annotations

import logging
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Iterable
from urllib.parse import quote

import pandas as pd


logger = logging.getLogger(__name__)

REQUIRED_UNIVERSE_COLUMNS = {"ticker", "name", "sector"}
PRICE_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]
HOLDING_COLUMNS = [
    "ticker",
    "name",
    "shares",
    "avg_cost",
    "asset_type",
    "trade_type",
    "thesis",
]


def load_universe(path: str | Path) -> pd.DataFrame:
    universe_path = Path(path).expanduser()
    if not universe_path.exists():
        raise FileNotFoundError(f"Universe file not found: {universe_path}")

    frame = pd.read_csv(universe_path)
    missing = REQUIRED_UNIVERSE_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"Universe file missing columns: {sorted(missing)}")

    result = frame.copy()
    result["ticker"] = result["ticker"].astype(str).str.strip().str.upper()
    result["name"] = result["name"].astype(str).str.strip()
    result["sector"] = result["sector"].astype(str).str.strip()
    for column in ["theme", "risk_bucket", "notes"]:
        if column not in result:
            result[column] = ""
        result[column] = result[column].fillna("").astype(str).str.strip()
    if "fundamental_score" not in result:
        result["fundamental_score"] = pd.NA
    result = result.drop_duplicates(subset=["ticker"]).reset_index(drop=True)
    if result.empty:
        raise ValueError("Universe file has no tickers.")
    return result


def load_holdings(path: str | Path) -> pd.DataFrame:
    holdings_path = Path(path).expanduser()
    if not holdings_path.exists():
        return pd.DataFrame(columns=HOLDING_COLUMNS)

    frame = pd.read_csv(holdings_path)
    required = {"ticker", "shares", "avg_cost"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Holdings file missing columns: {sorted(missing)}")

    result = frame.copy()
    for column in HOLDING_COLUMNS:
        if column not in result:
            result[column] = "" if column not in {"shares", "avg_cost"} else 0.0
    result["ticker"] = result["ticker"].astype(str).str.strip().str.upper()
    result["name"] = result["name"].fillna("").astype(str).str.strip()
    result["shares"] = pd.to_numeric(result["shares"], errors="coerce").fillna(0.0)
    result["avg_cost"] = pd.to_numeric(result["avg_cost"], errors="coerce").fillna(0.0)
    result["asset_type"] = result["asset_type"].fillna("stock").astype(str).str.lower().str.strip()
    result["trade_type"] = result["trade_type"].fillna("").astype(str).str.strip()
    result["thesis"] = result["thesis"].fillna("").astype(str).str.strip()
    result["is_cash"] = result["asset_type"].eq("cash") | result["ticker"].eq("CASH")
    keep = result["is_cash"] | result["shares"].ne(0)
    return result.loc[keep].reset_index(drop=True)


def fetch_price_history_cached(
    tickers: Iterable[str],
    lookback_days: int,
    as_of: date | None,
    cache_dir: str | Path,
    cache_enabled: bool = True,
    force_refresh: bool = False,
    max_stale_calendar_days: int = 5,
    min_coverage_ratio: float = 0.70,
    use_stale_on_failure: bool = True,
    batch_size: int = 40,
    pause_seconds: float = 1.0,
    max_retries: int = 3,
    retry_pause_seconds: float = 10.0,
    timeout_seconds: float = 45.0,
    threads: bool | int = True,
) -> dict[str, pd.DataFrame]:
    ticker_list = unique_tickers(tickers)
    if not ticker_list:
        return {}

    if not cache_enabled:
        return fetch_price_history(
            ticker_list,
            lookback_days=lookback_days,
            as_of=as_of,
            batch_size=batch_size,
            pause_seconds=pause_seconds,
            max_retries=max_retries,
            retry_pause_seconds=retry_pause_seconds,
            timeout_seconds=timeout_seconds,
            threads=threads,
        )

    cache_root = Path(cache_dir).expanduser()
    cache_root.mkdir(parents=True, exist_ok=True)
    end_date = as_of or date.today()
    result: dict[str, pd.DataFrame] = {}
    cached_frames: dict[str, pd.DataFrame] = {}
    to_fetch: list[str] = []

    for ticker in ticker_list:
        cached = read_price_cache(cache_root, ticker)
        if not cached.empty:
            cached_frames[ticker] = cached
        usable = (
            not force_refresh
            and price_cache_usable(
                cached,
                lookback_days=lookback_days,
                as_of=end_date,
                max_stale_calendar_days=max_stale_calendar_days,
                min_coverage_ratio=min_coverage_ratio,
            )
        )
        if usable:
            result[ticker] = trim_price_frame(cached, lookback_days, end_date)
        else:
            to_fetch.append(ticker)

    logger.info(
        "Price cache hits: %d/%d; downloading %d symbols",
        len(result),
        len(ticker_list),
        len(to_fetch),
    )

    downloaded: dict[str, pd.DataFrame] = {}
    if to_fetch:
        try:
            downloaded = fetch_price_history(
                to_fetch,
                lookback_days=lookback_days,
                as_of=as_of,
                batch_size=batch_size,
                pause_seconds=pause_seconds,
                max_retries=max_retries,
                retry_pause_seconds=retry_pause_seconds,
                timeout_seconds=timeout_seconds,
                threads=threads,
            )
        except Exception as exc:  # pragma: no cover - network dependent
            logger.warning("Price download failed; using stale cache where possible: %s", exc)

    for ticker, frame in downloaded.items():
        cached = cached_frames.get(ticker, pd.DataFrame())
        merged = merge_price_frames(cached, frame)
        if not merged.empty:
            write_price_cache(cache_root, ticker, merged)
            result[ticker] = trim_price_frame(merged, lookback_days, end_date)

    if use_stale_on_failure:
        for ticker in to_fetch:
            if ticker not in result and ticker in cached_frames:
                stale = trim_price_frame(cached_frames[ticker], lookback_days, end_date)
                if not stale.empty:
                    result[ticker] = stale

    return result


def fetch_price_history(
    tickers: Iterable[str],
    lookback_days: int,
    as_of: date | None,
    batch_size: int = 40,
    pause_seconds: float = 1.0,
    max_retries: int = 3,
    retry_pause_seconds: float = 10.0,
    timeout_seconds: float = 45.0,
    threads: bool | int = True,
) -> dict[str, pd.DataFrame]:
    ticker_list = unique_tickers(tickers)
    if not ticker_list:
        return {}

    import yfinance as yf

    end_date = as_of or date.today()
    calendar_days = max(int(lookback_days * 2.2), lookback_days + 90)
    start_date = end_date - timedelta(days=calendar_days)
    result: dict[str, pd.DataFrame] = {}
    batches = list(chunked(ticker_list, max(1, int(batch_size))))

    for batch_index, batch in enumerate(batches, start=1):
        batch_result: dict[str, pd.DataFrame] = {}
        for attempt in range(max(0, int(max_retries)) + 1):
            try:
                logger.info(
                    "Downloading price batch %d/%d (%d symbols)",
                    batch_index,
                    len(batches),
                    len(batch),
                )
                raw = yf.download(
                    tickers=batch if len(batch) > 1 else batch[0],
                    start=start_date.isoformat(),
                    end=(end_date + timedelta(days=1)).isoformat(),
                    auto_adjust=True,
                    group_by="ticker",
                    progress=False,
                    threads=threads,
                    timeout=max(1.0, float(timeout_seconds)),
                )
                batch_result = split_yfinance_download(raw, batch, lookback_days)
                break
            except Exception as exc:  # pragma: no cover - network dependent
                if attempt >= max_retries:
                    logger.warning("Price batch failed after retries: %s", exc)
                    break
                sleep_seconds = max(0.0, float(retry_pause_seconds)) * (attempt + 1)
                logger.warning("Price batch attempt failed: %s; retrying in %.1fs", exc, sleep_seconds)
                time.sleep(sleep_seconds)

        result.update(batch_result)
        if pause_seconds > 0 and batch_index < len(batches):
            time.sleep(float(pause_seconds))

    return result


def split_yfinance_download(raw: pd.DataFrame, tickers: list[str], lookback_days: int) -> dict[str, pd.DataFrame]:
    if raw is None or raw.empty:
        return {}

    result: dict[str, pd.DataFrame] = {}
    if isinstance(raw.columns, pd.MultiIndex):
        level0 = set(str(value).upper() for value in raw.columns.get_level_values(0))
        level1 = set(str(value).upper() for value in raw.columns.get_level_values(1))
        for ticker in tickers:
            frame = pd.DataFrame()
            if ticker.upper() in level0:
                try:
                    frame = raw[ticker]
                except KeyError:
                    frame = raw.xs(ticker, level=0, axis=1, drop_level=True)
            elif ticker.upper() in level1:
                frame = raw.xs(ticker, level=1, axis=1, drop_level=True)
            normalized = normalize_price_frame(frame, lookback_days)
            if not normalized.empty:
                result[ticker] = normalized
    else:
        normalized = normalize_price_frame(raw, lookback_days)
        if normalized.empty:
            return {}
        if len(tickers) == 1:
            result[tickers[0]] = normalized
    return result


def normalize_price_frame(frame: pd.DataFrame, lookback_days: int | None = None) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame(columns=PRICE_COLUMNS)

    result = frame.copy()
    if "Date" in result.columns:
        result["Date"] = pd.to_datetime(result["Date"], errors="coerce")
        result = result.set_index("Date")
    result.index = pd.to_datetime(result.index, errors="coerce")
    result = result.loc[result.index.notna()].sort_index()
    if "Close" not in result and "Adj Close" in result:
        result["Close"] = result["Adj Close"]
    for column in PRICE_COLUMNS:
        if column not in result:
            result[column] = 0.0
        result[column] = pd.to_numeric(result[column], errors="coerce")
    result = result.loc[:, PRICE_COLUMNS].dropna(subset=["Close"])
    result = result[~result.index.duplicated(keep="last")]
    if lookback_days:
        result = result.tail(int(lookback_days))
    result.index.name = "Date"
    return result


def read_price_cache(cache_dir: Path, ticker: str) -> pd.DataFrame:
    path = cache_path(cache_dir, ticker)
    if not path.exists():
        return pd.DataFrame(columns=PRICE_COLUMNS)
    try:
        return normalize_price_frame(pd.read_csv(path), None)
    except Exception:
        return pd.DataFrame(columns=PRICE_COLUMNS)


def write_price_cache(cache_dir: Path, ticker: str, frame: pd.DataFrame) -> None:
    normalized = normalize_price_frame(frame, None)
    if normalized.empty:
        return
    cache_dir.mkdir(parents=True, exist_ok=True)
    normalized.reset_index().to_csv(cache_path(cache_dir, ticker), index=False)


def cache_path(cache_dir: Path, ticker: str) -> Path:
    return cache_dir / f"{quote(str(ticker).upper(), safe='')}.csv"


def price_cache_usable(
    frame: pd.DataFrame,
    lookback_days: int,
    as_of: date,
    max_stale_calendar_days: int,
    min_coverage_ratio: float,
) -> bool:
    if frame.empty:
        return False
    trimmed = trim_price_frame(frame, lookback_days, as_of)
    min_rows = max(10, int(lookback_days * float(min_coverage_ratio)))
    if len(trimmed) < min_rows:
        return False
    latest = trimmed.index.max().date()
    if latest > as_of:
        return False
    return (as_of - latest).days <= int(max_stale_calendar_days)


def trim_price_frame(frame: pd.DataFrame, lookback_days: int, as_of: date) -> pd.DataFrame:
    if frame.empty:
        return frame
    normalized = normalize_price_frame(frame, None)
    normalized = normalized[normalized.index.date <= as_of]
    return normalized.tail(int(lookback_days))


def merge_price_frames(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    frames = [normalize_price_frame(frame, None) for frame in [left, right] if frame is not None and not frame.empty]
    if not frames:
        return pd.DataFrame(columns=PRICE_COLUMNS)
    merged = pd.concat(frames).sort_index()
    merged = merged[~merged.index.duplicated(keep="last")]
    return normalize_price_frame(merged, None)


def unique_tickers(tickers: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(str(ticker).strip().upper() for ticker in tickers if str(ticker).strip()))


def chunked(values: list[str], size: int) -> Iterable[list[str]]:
    for index in range(0, len(values), size):
        yield values[index : index + size]

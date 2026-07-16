from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

from .config import resolve_path
from .data import fetch_price_history_cached, load_universe, unique_tickers
from .indicators import is_finite


@dataclass
class EventStudyResult:
    as_of: date
    universe_count: int
    scanned_count: int
    event_count: int
    observation_count: int
    event_rate: float
    events: pd.DataFrame
    observations: pd.DataFrame
    factor_lift: pd.DataFrame
    sector_summary: pd.DataFrame
    regime_summary: pd.DataFrame
    current_watchlist: pd.DataFrame
    report_text: str


def run_event_study(
    config: dict[str, Any],
    config_dir: Path,
    as_of: date,
    *,
    lookback_days: int | None = None,
    horizon_days: int | None = None,
    min_forward_return: float | None = None,
    min_forward_excess_return: float | None = None,
    min_avg_dollar_volume: float | None = None,
    sample_step_days: int | None = None,
    max_symbols: int | None = None,
    selection_scores_path: str | Path | None = None,
    force_refresh: bool = False,
) -> EventStudyResult:
    study_cfg = dict(config.get("event_study", {}))
    universe_path = resolve_path(config.get("universe_path", "data/universe_us_all.csv"), config_dir)
    universe = load_universe(universe_path)

    lookback = int(lookback_days or study_cfg.get("lookback_days", 1600))
    horizon = int(horizon_days or study_cfg.get("horizon_days", 60))
    min_return = float(min_forward_return if min_forward_return is not None else study_cfg.get("min_forward_return", 0.30))
    min_excess = float(
        min_forward_excess_return
        if min_forward_excess_return is not None
        else study_cfg.get("min_forward_excess_return", 0.15)
    )
    min_liquidity = float(
        min_avg_dollar_volume if min_avg_dollar_volume is not None else study_cfg.get("min_avg_dollar_volume", 25_000_000)
    )
    sample_step = max(1, int(sample_step_days or study_cfg.get("sample_step_days", 5)))

    selected_universe = select_universe(
        universe,
        selection_scores_path=selection_scores_path or study_cfg.get("selection_scores_path"),
        max_symbols=max_symbols if max_symbols is not None else study_cfg.get("max_symbols"),
        min_avg_dollar_volume=min_liquidity,
    )
    benchmark = str(config.get("market", {}).get("benchmark", "SPY")).upper()
    symbols = unique_tickers([benchmark, *selected_universe["ticker"].tolist()])

    data_fetch_cfg = config.get("data_fetch", {})
    cache_cfg = data_fetch_cfg.get("cache", {})
    price_data = fetch_price_history_cached(
        symbols,
        lookback_days=lookback,
        as_of=as_of,
        cache_dir=resolve_path(cache_cfg.get("dir", "data/price_cache"), config_dir),
        cache_enabled=bool(cache_cfg.get("enabled", True)),
        force_refresh=force_refresh or bool(cache_cfg.get("force_refresh", False)),
        max_stale_calendar_days=int(cache_cfg.get("max_stale_calendar_days", 5)),
        min_coverage_ratio=float(cache_cfg.get("min_coverage_ratio", 0.70)),
        use_stale_on_failure=bool(cache_cfg.get("use_stale_on_failure", True)),
        batch_size=int(data_fetch_cfg.get("batch_size", 40)),
        pause_seconds=float(data_fetch_cfg.get("pause_seconds", 1.0)),
        max_retries=int(data_fetch_cfg.get("max_retries", 3)),
        retry_pause_seconds=float(data_fetch_cfg.get("retry_pause_seconds", 10.0)),
        timeout_seconds=float(data_fetch_cfg.get("timeout_seconds", 45.0)),
        threads=data_fetch_cfg.get("threads", True),
    )

    benchmark_frame = price_data.get(benchmark, pd.DataFrame())
    benchmark_features = benchmark_history_features(benchmark_frame, horizon)
    observations = build_observations(
        selected_universe,
        price_data,
        benchmark_features,
        benchmark,
        horizon=horizon,
        min_forward_return=min_return,
        min_forward_excess_return=min_excess,
        min_avg_dollar_volume=min_liquidity,
        sample_step_days=sample_step,
    )
    current_observations = build_current_observations(
        selected_universe,
        price_data,
        benchmark_features,
        benchmark,
        horizon=horizon,
        min_avg_dollar_volume=min_liquidity,
    )
    if observations.empty:
        empty = pd.DataFrame()
        return EventStudyResult(
            as_of=as_of,
            universe_count=len(universe),
            scanned_count=max(0, len(symbols) - 1),
            event_count=0,
            observation_count=0,
            event_rate=math.nan,
            events=empty,
            observations=empty,
            factor_lift=empty,
            sector_summary=empty,
            regime_summary=empty,
            current_watchlist=empty,
            report_text="No usable event-study observations.",
        )

    observations = add_sector_context(observations)
    observations = add_factor_flags(observations)
    events = extract_event_episodes(observations, horizon)
    factor_lift = build_factor_lift(observations)
    sector_summary = build_sector_summary(observations, events)
    regime_summary = build_regime_summary(observations)
    current_observations = add_sector_context(current_observations)
    current_observations = add_factor_flags(current_observations)
    current_watchlist = build_current_factor_watchlist(current_observations, factor_lift)
    event_rate = float(observations["is_big_rally"].mean()) if len(observations) else math.nan
    report_text = render_event_study_report(
        as_of=as_of,
        lookback_days=lookback,
        horizon_days=horizon,
        min_forward_return=min_return,
        min_forward_excess_return=min_excess,
        universe_count=len(universe),
        scanned_count=max(0, len(symbols) - 1),
        observations=observations,
        events=events,
        factor_lift=factor_lift,
        sector_summary=sector_summary,
        regime_summary=regime_summary,
        current_watchlist=current_watchlist,
    )

    return EventStudyResult(
        as_of=as_of,
        universe_count=len(universe),
        scanned_count=max(0, len(symbols) - 1),
        event_count=len(events),
        observation_count=len(observations),
        event_rate=event_rate,
        events=events,
        observations=observations,
        factor_lift=factor_lift,
        sector_summary=sector_summary,
        regime_summary=regime_summary,
        current_watchlist=current_watchlist,
        report_text=report_text,
    )


def select_universe(
    universe: pd.DataFrame,
    *,
    selection_scores_path: str | Path | None,
    max_symbols: int | str | None,
    min_avg_dollar_volume: float,
) -> pd.DataFrame:
    result = universe.copy()
    max_count = int(max_symbols) if max_symbols not in {None, "", 0, "0"} else None
    if selection_scores_path:
        score_path = Path(selection_scores_path).expanduser()
        if score_path.exists():
            scores = pd.read_csv(score_path)
            if {"ticker", "avg_dollar_volume_20d"}.issubset(scores.columns):
                scores = scores.copy()
                scores["ticker"] = scores["ticker"].astype(str).str.upper()
                scores["avg_dollar_volume_20d"] = pd.to_numeric(
                    scores["avg_dollar_volume_20d"],
                    errors="coerce",
                )
                result = result.merge(
                    scores[["ticker", "avg_dollar_volume_20d"]],
                    on="ticker",
                    how="left",
                )
                result = result[pd.to_numeric(result["avg_dollar_volume_20d"], errors="coerce") >= min_avg_dollar_volume]
                result = result.sort_values("avg_dollar_volume_20d", ascending=False)

    if max_count is not None and len(result) > max_count:
        result = result.head(max_count)
    return result.reset_index(drop=True)


def benchmark_history_features(frame: pd.DataFrame, horizon: int) -> pd.DataFrame:
    if frame is None or frame.empty or "Close" not in frame:
        return pd.DataFrame(columns=["date", "benchmark_forward_return"])
    close = pd.to_numeric(frame.sort_index()["Close"], errors="coerce").dropna()
    future_max = close.shift(-1).rolling(horizon, min_periods=max(5, horizon // 3)).max().shift(-(horizon - 1))
    result = pd.DataFrame(
        {
            "date": close.index.normalize(),
            "benchmark_close": close.values,
            "benchmark_forward_return": future_max.values / close.values - 1,
            "benchmark_return_20d": close.pct_change(20).values,
            "benchmark_return_60d": close.pct_change(60).values,
        }
    )
    return result.dropna(subset=["date"]).drop_duplicates("date")


def build_observations(
    universe: pd.DataFrame,
    price_data: dict[str, pd.DataFrame],
    benchmark_features: pd.DataFrame,
    benchmark: str,
    *,
    horizon: int,
    min_forward_return: float,
    min_forward_excess_return: float,
    min_avg_dollar_volume: float,
    sample_step_days: int,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    benchmark_by_date = benchmark_features.set_index("date") if not benchmark_features.empty else pd.DataFrame()
    meta = universe.set_index("ticker").to_dict("index") if not universe.empty else {}

    for ticker in universe["ticker"].astype(str).str.upper().tolist():
        if ticker == benchmark:
            continue
        frame = price_data.get(ticker, pd.DataFrame())
        features = symbol_history_features(frame, horizon, require_forward=True)
        if features.empty:
            continue
        features["ticker"] = ticker
        features["name"] = str(meta.get(ticker, {}).get("name", ticker))
        features["sector"] = str(meta.get(ticker, {}).get("sector", ""))
        features["theme"] = str(meta.get(ticker, {}).get("theme", ""))
        features["risk_bucket"] = str(meta.get(ticker, {}).get("risk_bucket", ""))
        if not benchmark_by_date.empty:
            features = features.merge(
                benchmark_by_date[
                    ["benchmark_forward_return", "benchmark_return_20d", "benchmark_return_60d"]
                ].reset_index(),
                on="date",
                how="left",
            )
        else:
            features["benchmark_forward_return"] = math.nan
            features["benchmark_return_20d"] = math.nan
            features["benchmark_return_60d"] = math.nan

        features["forward_excess_return"] = features["forward_max_return"] - features["benchmark_forward_return"]
        features["relative_return_20d"] = features["return_20d"] - features["benchmark_return_20d"]
        features["relative_return_60d"] = features["return_60d"] - features["benchmark_return_60d"]
        features["is_big_rally"] = (
            (features["forward_max_return"] >= min_forward_return)
            & (features["forward_excess_return"] >= min_forward_excess_return)
        )
        features = features[pd.to_numeric(features["avg_dollar_volume_20d"], errors="coerce") >= min_avg_dollar_volume]
        features = features.iloc[::sample_step_days].copy()
        if not features.empty:
            rows.append(features)

    if not rows:
        return pd.DataFrame()
    result = pd.concat(rows, ignore_index=True)
    result["is_big_rally"] = result["is_big_rally"].fillna(False).astype(bool)
    return result.sort_values(["date", "ticker"]).reset_index(drop=True)


def build_current_observations(
    universe: pd.DataFrame,
    price_data: dict[str, pd.DataFrame],
    benchmark_features: pd.DataFrame,
    benchmark: str,
    *,
    horizon: int,
    min_avg_dollar_volume: float,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    benchmark_by_date = benchmark_features.set_index("date") if not benchmark_features.empty else pd.DataFrame()
    meta = universe.set_index("ticker").to_dict("index") if not universe.empty else {}

    for ticker in universe["ticker"].astype(str).str.upper().tolist():
        if ticker == benchmark:
            continue
        frame = price_data.get(ticker, pd.DataFrame())
        features = symbol_history_features(frame, horizon, require_forward=False)
        if features.empty:
            continue
        latest = features.tail(1).copy()
        latest["ticker"] = ticker
        latest["name"] = str(meta.get(ticker, {}).get("name", ticker))
        latest["sector"] = str(meta.get(ticker, {}).get("sector", ""))
        latest["theme"] = str(meta.get(ticker, {}).get("theme", ""))
        latest["risk_bucket"] = str(meta.get(ticker, {}).get("risk_bucket", ""))
        if not benchmark_by_date.empty:
            latest = latest.merge(
                benchmark_by_date[["benchmark_return_20d", "benchmark_return_60d"]].reset_index(),
                on="date",
                how="left",
            )
        else:
            latest["benchmark_return_20d"] = math.nan
            latest["benchmark_return_60d"] = math.nan
        latest["benchmark_forward_return"] = math.nan
        latest["forward_excess_return"] = math.nan
        latest["relative_return_20d"] = latest["return_20d"] - latest["benchmark_return_20d"]
        latest["relative_return_60d"] = latest["return_60d"] - latest["benchmark_return_60d"]
        latest["is_big_rally"] = False
        latest = latest[pd.to_numeric(latest["avg_dollar_volume_20d"], errors="coerce") >= min_avg_dollar_volume]
        if not latest.empty:
            rows.append(latest)

    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True).sort_values(["date", "ticker"]).reset_index(drop=True)


def symbol_history_features(frame: pd.DataFrame, horizon: int, require_forward: bool = True) -> pd.DataFrame:
    if frame is None or frame.empty or "Close" not in frame:
        return pd.DataFrame()
    prices = frame.copy().sort_index()
    close = pd.to_numeric(prices["Close"], errors="coerce")
    high = pd.to_numeric(prices.get("High", close), errors="coerce")
    low = pd.to_numeric(prices.get("Low", close), errors="coerce")
    volume = pd.to_numeric(prices.get("Volume", pd.Series(index=prices.index, dtype=float)), errors="coerce").fillna(0)
    data = pd.DataFrame({"Close": close, "High": high, "Low": low, "Volume": volume}).dropna(subset=["Close"])
    if len(data) < max(260, horizon + 120):
        return pd.DataFrame()

    close = data["Close"]
    high = data["High"].fillna(close)
    low = data["Low"].fillna(close)
    volume = data["Volume"].fillna(0)
    previous_close = close.shift(1)
    true_range = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    future_max = close.shift(-1).rolling(horizon, min_periods=max(5, horizon // 3)).max().shift(-(horizon - 1))
    future_min = close.shift(-1).rolling(horizon, min_periods=max(5, horizon // 3)).min().shift(-(horizon - 1))

    ma20 = close.rolling(20).mean()
    ma50 = close.rolling(50).mean()
    ma200 = close.rolling(200).mean()
    high20_prev = high.shift(1).rolling(20).max()
    high60_prev = high.shift(1).rolling(60).max()
    high252_prev = high.shift(1).rolling(252).max()
    low60_prev = low.shift(1).rolling(60).min()
    avg_volume20 = volume.rolling(20).mean()
    avg_volume5 = volume.rolling(5).mean()
    avg_dollar_volume20 = avg_volume20 * close
    atr14 = true_range.rolling(14).mean()
    volatility20 = close.pct_change().rolling(20).std()
    rsi14 = rolling_rsi(close, 14)

    result = pd.DataFrame(
        {
            "date": close.index.normalize(),
            "close": close,
            "forward_max_return": future_max / close - 1,
            "forward_min_return": future_min / close - 1,
            "return_1d": close.pct_change(1),
            "return_5d": close.pct_change(5),
            "return_10d": close.pct_change(10),
            "return_20d": close.pct_change(20),
            "return_60d": close.pct_change(60),
            "return_120d": close.pct_change(120),
            "ma20": ma20,
            "ma50": ma50,
            "ma200": ma200,
            "pct_from_ma20": close / ma20 - 1,
            "pct_from_ma50": close / ma50 - 1,
            "pct_from_ma200": close / ma200 - 1,
            "ma20_slope_10d": ma20 / ma20.shift(10) - 1,
            "ma50_slope_20d": ma50 / ma50.shift(20) - 1,
            "drawdown_from_20d_high": close / high20_prev - 1,
            "drawdown_from_60d_high": close / high60_prev - 1,
            "drawdown_from_252d_high": close / high252_prev - 1,
            "pct_above_60d_low": close / low60_prev - 1,
            "volume_ratio_20d": volume / avg_volume20,
            "volume_5d_vs_20d": avg_volume5 / avg_volume20,
            "avg_dollar_volume_20d": avg_dollar_volume20,
            "atr_pct": atr14 / close,
            "volatility_20d": volatility20,
            "rsi14": rsi14,
        }
    )
    required = ["return_20d", "ma50", "avg_dollar_volume_20d"]
    if require_forward:
        required.append("forward_max_return")
    result = result.dropna(subset=required)
    return result.reset_index(drop=True)


def rolling_rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    rs = gain / loss.replace(0, pd.NA)
    return 100 - (100 / (1 + rs))


def add_sector_context(observations: pd.DataFrame) -> pd.DataFrame:
    result = observations.copy()
    if result.empty:
        return result
    grouped = result.groupby(["date", "sector"], dropna=False)
    sector_returns = grouped.agg(
        sector_return_20d=("return_20d", "median"),
        sector_return_60d=("return_60d", "median"),
        sector_member_count=("ticker", "count"),
    ).reset_index()
    sector_returns["sector_return_20d_rank_pct"] = sector_returns.groupby("date")["sector_return_20d"].rank(
        pct=True,
        ascending=False,
    )
    sector_returns["sector_return_60d_rank_pct"] = sector_returns.groupby("date")["sector_return_60d"].rank(
        pct=True,
        ascending=False,
    )
    result = result.merge(sector_returns, on=["date", "sector"], how="left")
    return result


def add_factor_flags(observations: pd.DataFrame) -> pd.DataFrame:
    result = observations.copy()
    result["above_ma50"] = result["pct_from_ma50"] > 0
    result["above_ma200"] = result["pct_from_ma200"] > 0
    result["ma20_gt_ma50"] = result["ma20"] > result["ma50"]
    result["ma50_gt_ma200"] = result["ma50"] > result["ma200"]
    result["positive_ma20_slope"] = result["ma20_slope_10d"] > 0
    result["positive_ma50_slope"] = result["ma50_slope_20d"] > 0
    result["relative_strength_20d"] = result["relative_return_20d"] >= 0.05
    result["relative_strength_60d"] = result["relative_return_60d"] >= 0.10
    result["near_60d_high"] = result["drawdown_from_60d_high"] >= -0.05
    result["near_252d_high"] = result["drawdown_from_252d_high"] >= -0.08
    result["new_20d_high"] = result["drawdown_from_20d_high"] >= 0
    result["healthy_pullback_to_ma20"] = result["pct_from_ma20"].between(-0.05, 0.03) & result["above_ma50"]
    result["volume_expansion"] = result["volume_ratio_20d"] >= 1.5
    result["volume_dryup"] = result["volume_5d_vs_20d"] <= 0.75
    result["rsi_constructive"] = result["rsi14"].between(45, 70)
    result["rsi_hot"] = result["rsi14"] > 70
    result["low_atr"] = result["atr_pct"] <= 0.035
    result["high_atr"] = result["atr_pct"] >= 0.06
    result["sector_top_quartile_20d"] = result["sector_return_20d_rank_pct"] <= 0.25
    result["sector_top_quartile_60d"] = result["sector_return_60d_rank_pct"] <= 0.25
    result["sector_rotation_plus_stock_rs"] = result["sector_top_quartile_20d"] & result["relative_strength_20d"]
    result["breakout_with_volume"] = result["new_20d_high"] & result["volume_expansion"] & result["relative_strength_20d"]
    result["quiet_base_near_high"] = result["volume_dryup"] & result["near_60d_high"] & result["above_ma50"]
    result["uptrend_pullback"] = (
        result["ma50_gt_ma200"]
        & result["positive_ma50_slope"]
        & result["healthy_pullback_to_ma20"]
        & result["rsi_constructive"]
    )
    result["deep_reversal_risk"] = (result["drawdown_from_252d_high"] <= -0.30) & result["volume_expansion"]
    return result


def extract_event_episodes(observations: pd.DataFrame, horizon: int) -> pd.DataFrame:
    if observations.empty:
        return pd.DataFrame()
    event_rows = []
    for ticker, frame in observations[observations["is_big_rally"]].sort_values(["ticker", "date"]).groupby("ticker"):
        last_event_date: pd.Timestamp | None = None
        for row in frame.itertuples(index=False):
            current_date = pd.Timestamp(row.date)
            if last_event_date is not None and (current_date - last_event_date).days < int(horizon * 1.4):
                continue
            record = row._asdict()
            record["event_tag"] = classify_event_tag(record)
            event_rows.append(record)
            last_event_date = current_date
    if not event_rows:
        return pd.DataFrame()
    events = pd.DataFrame(event_rows)
    return events.sort_values(["forward_max_return", "forward_excess_return"], ascending=False).reset_index(drop=True)


def classify_event_tag(row: dict[str, Any]) -> str:
    if bool(row.get("breakout_with_volume")):
        return "量價突破"
    if bool(row.get("sector_rotation_plus_stock_rs")):
        return "產業輪動 + 個股相對強"
    if bool(row.get("uptrend_pullback")):
        return "上升趨勢回檔"
    if bool(row.get("quiet_base_near_high")):
        return "高位安靜整理"
    if bool(row.get("deep_reversal_risk")):
        return "深跌反轉/軋空"
    return "混合/未分類"


def build_factor_lift(observations: pd.DataFrame) -> pd.DataFrame:
    factor_columns = [
        "above_ma50",
        "above_ma200",
        "ma20_gt_ma50",
        "ma50_gt_ma200",
        "positive_ma20_slope",
        "positive_ma50_slope",
        "relative_strength_20d",
        "relative_strength_60d",
        "near_60d_high",
        "near_252d_high",
        "new_20d_high",
        "healthy_pullback_to_ma20",
        "volume_expansion",
        "volume_dryup",
        "rsi_constructive",
        "rsi_hot",
        "low_atr",
        "high_atr",
        "sector_top_quartile_20d",
        "sector_top_quartile_60d",
        "sector_rotation_plus_stock_rs",
        "breakout_with_volume",
        "quiet_base_near_high",
        "uptrend_pullback",
        "deep_reversal_risk",
    ]
    base_rate = float(observations["is_big_rally"].mean()) if len(observations) else math.nan
    rows: list[dict[str, Any]] = []
    for factor in factor_columns:
        if factor not in observations:
            continue
        subset = observations[observations[factor].fillna(False).astype(bool)]
        if subset.empty:
            continue
        event_rate = float(subset["is_big_rally"].mean())
        rows.append(
            {
                "factor": factor,
                "observations": len(subset),
                "events": int(subset["is_big_rally"].sum()),
                "event_rate": event_rate,
                "base_event_rate": base_rate,
                "lift": event_rate / base_rate if is_finite(base_rate) and base_rate > 0 else math.nan,
                "avg_forward_max_return": float(subset["forward_max_return"].mean()),
                "median_forward_max_return": float(subset["forward_max_return"].median()),
            }
        )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["lift", "events"], ascending=False).reset_index(drop=True)


def build_sector_summary(observations: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    obs = observations.groupby("sector", dropna=False).agg(
        observations=("ticker", "count"),
        labeled_events=("is_big_rally", "sum"),
        avg_forward_max_return=("forward_max_return", "mean"),
        median_forward_max_return=("forward_max_return", "median"),
        avg_relative_return_20d=("relative_return_20d", "mean"),
    )
    episodes = events.groupby("sector", dropna=False).agg(event_episodes=("ticker", "count")) if not events.empty else pd.DataFrame()
    result = obs.join(episodes, how="left").fillna({"event_episodes": 0})
    result["event_rate"] = result["labeled_events"] / result["observations"]
    result["event_episodes"] = result["event_episodes"].astype(int)
    return result.reset_index().sort_values(["event_rate", "event_episodes"], ascending=False).reset_index(drop=True)


def build_regime_summary(observations: pd.DataFrame) -> pd.DataFrame:
    result = observations.copy()
    result["benchmark_regime"] = "neutral"
    result.loc[result["benchmark_return_60d"] >= 0.05, "benchmark_regime"] = "market_uptrend"
    result.loc[result["benchmark_return_60d"] <= -0.05, "benchmark_regime"] = "market_downtrend"
    result["volatility_regime"] = "normal_vol"
    vol_q75 = result["volatility_20d"].quantile(0.75)
    vol_q25 = result["volatility_20d"].quantile(0.25)
    result.loc[result["volatility_20d"] >= vol_q75, "volatility_regime"] = "high_vol"
    result.loc[result["volatility_20d"] <= vol_q25, "volatility_regime"] = "low_vol"
    summary = result.groupby(["benchmark_regime", "volatility_regime"], dropna=False).agg(
        observations=("ticker", "count"),
        events=("is_big_rally", "sum"),
        avg_forward_max_return=("forward_max_return", "mean"),
    )
    summary["event_rate"] = summary["events"] / summary["observations"]
    return summary.reset_index().sort_values("event_rate", ascending=False).reset_index(drop=True)


def build_current_factor_watchlist(observations: pd.DataFrame, factor_lift: pd.DataFrame) -> pd.DataFrame:
    if observations.empty or factor_lift.empty:
        return pd.DataFrame()

    useful = factor_lift[
        (pd.to_numeric(factor_lift["lift"], errors="coerce") >= 1.20)
        & (pd.to_numeric(factor_lift["events"], errors="coerce") >= 100)
    ].copy()
    if useful.empty:
        return pd.DataFrame()

    latest = observations.sort_values(["ticker", "date"]).groupby("ticker", as_index=False).tail(1).copy()
    rows: list[dict[str, Any]] = []
    for row in latest.itertuples(index=False):
        row_dict = row._asdict()
        active: list[str] = []
        score = 0.0
        for factor_row in useful.itertuples(index=False):
            factor = str(factor_row.factor)
            if bool(row_dict.get(factor, False)):
                active.append(factor)
                lift = float(factor_row.lift)
                score += max(0.0, math.log(lift)) * 100.0
        if score <= 0:
            continue
        setup = current_setup_type(row_dict)
        rows.append(
            {
                "date": row_dict.get("date"),
                "ticker": row_dict.get("ticker"),
                "name": row_dict.get("name"),
                "sector": row_dict.get("sector"),
                "close": row_dict.get("close"),
                "factor_score": round(score, 2),
                "actionability_score": round(score * actionability_multiplier(row_dict), 2),
                "risk_tier": current_risk_tier(row_dict),
                "setup_type": setup,
                "active_factor_count": len(active),
                "active_factors": ", ".join(human_factor(factor) for factor in active),
                "return_20d": row_dict.get("return_20d"),
                "return_60d": row_dict.get("return_60d"),
                "relative_return_20d": row_dict.get("relative_return_20d"),
                "relative_return_60d": row_dict.get("relative_return_60d"),
                "sector_return_20d": row_dict.get("sector_return_20d"),
                "volume_ratio_20d": row_dict.get("volume_ratio_20d"),
                "volume_5d_vs_20d": row_dict.get("volume_5d_vs_20d"),
                "rsi14": row_dict.get("rsi14"),
                "atr_pct": row_dict.get("atr_pct"),
                "drawdown_from_60d_high": row_dict.get("drawdown_from_60d_high"),
                "drawdown_from_252d_high": row_dict.get("drawdown_from_252d_high"),
            }
        )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(
        ["actionability_score", "factor_score", "relative_return_20d"],
        ascending=False,
    ).reset_index(drop=True)


def actionability_multiplier(row: dict[str, Any]) -> float:
    multiplier = 1.0
    if bool(row.get("deep_reversal_risk")):
        multiplier *= 0.45
    if bool(row.get("high_atr")):
        multiplier *= 0.65
    if safe_float(row.get("rsi14")) > 80:
        multiplier *= 0.75
    if safe_float(row.get("drawdown_from_252d_high")) <= -0.60:
        multiplier *= 0.70
    return multiplier


def current_risk_tier(row: dict[str, Any]) -> str:
    if bool(row.get("deep_reversal_risk")) or bool(row.get("high_atr")):
        return "speculative"
    if bool(row.get("breakout_with_volume")) or bool(row.get("sector_rotation_plus_stock_rs")):
        return "active"
    return "watch"


def current_setup_type(row: dict[str, Any]) -> str:
    if bool(row.get("deep_reversal_risk")):
        return "深跌反轉/軋空，僅小倉"
    if bool(row.get("breakout_with_volume")):
        return "放量突破"
    if bool(row.get("sector_rotation_plus_stock_rs")):
        return "產業輪動 + 個股相對強"
    if bool(row.get("uptrend_pullback")):
        return "上升趨勢回檔"
    if bool(row.get("quiet_base_near_high")):
        return "高位安靜整理"
    if bool(row.get("relative_strength_60d")):
        return "中期相對強勢"
    return "早期因子累積"


def safe_float(value: Any, default: float = math.nan) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return default
    return numeric if math.isfinite(numeric) else default


def render_event_study_report(
    *,
    as_of: date,
    lookback_days: int,
    horizon_days: int,
    min_forward_return: float,
    min_forward_excess_return: float,
    universe_count: int,
    scanned_count: int,
    observations: pd.DataFrame,
    events: pd.DataFrame,
    factor_lift: pd.DataFrame,
    sector_summary: pd.DataFrame,
    regime_summary: pd.DataFrame,
    current_watchlist: pd.DataFrame,
) -> str:
    base_rate = float(observations["is_big_rally"].mean()) if len(observations) else math.nan
    lines = [
        f"# 美股波段大漲事件研究 ({as_of.isoformat()})",
        "",
        "## 研究定義",
        f"- 樣本 universe：{universe_count:,} 檔；本次掃描：{scanned_count:,} 檔。",
        f"- 回看資料：約 {lookback_days:,} 個交易日；觀察筆數：{len(observations):,}。",
        f"- 大漲事件：未來 {horizon_days} 個交易日最大漲幅 >= {min_forward_return:.0%}，且相對 SPY >= {min_forward_excess_return:.0%}。",
        f"- 樣本基準事件率：{base_rate:.2%}。",
        "",
        "## 最有用的提前因子",
    ]
    for row in factor_lift.head(12).itertuples(index=False):
        lines.append(
            f"- {human_factor(row.factor)}：事件率 {row.event_rate:.2%}，lift {row.lift:.2f}x，事件 {int(row.events):,} / 樣本 {int(row.observations):,}。"
        )

    lines.extend(["", "## 產業分布"])
    for row in sector_summary.head(10).itertuples(index=False):
        lines.append(
            f"- {row.sector}：事件率 {row.event_rate:.2%}，事件段 {int(row.event_episodes):,}，觀察 {int(row.observations):,}。"
        )

    lines.extend(["", "## 常見大漲型態"])
    if events.empty or "event_tag" not in events:
        lines.append("- 無事件段。")
    else:
        tag_counts = events["event_tag"].value_counts()
        for tag, count in tag_counts.items():
            lines.append(f"- {tag}：{int(count):,} 段。")

    lines.extend(["", "## 市場環境"])
    for row in regime_summary.head(8).itertuples(index=False):
        lines.append(
            f"- {row.benchmark_regime} / {row.volatility_regime}：事件率 {row.event_rate:.2%}，事件 {int(row.events):,} / 樣本 {int(row.observations):,}。"
        )

    lines.extend(
        [
            "",
            "## 目前開始出現早期訊號的標的",
        ]
    )
    if current_watchlist.empty:
        lines.append("- 目前沒有符合高 lift 因子組合的流動性標的。")
    else:
        actionable = current_watchlist[current_watchlist["risk_tier"].ne("speculative")].copy()
        speculative = current_watchlist[current_watchlist["risk_tier"].eq("speculative")].copy()
        lines.append("### 可操作趨勢型")
        if actionable.empty:
            lines.append("- 目前沒有乾淨的趨勢型早期訊號。")
        for row in actionable.head(10).itertuples(index=False):
            lines.append(
                f"- {row.ticker} {row.name}：{row.setup_type}，因子分 {row.factor_score:.1f}，"
                f"可操作分 {row.actionability_score:.1f}，20日相對 {row.relative_return_20d:.1%}，量比 {row.volume_ratio_20d:.2f}。"
            )
        lines.append("### 高波動事件型")
        if speculative.empty:
            lines.append("- 目前沒有高波動事件型訊號。")
        for row in speculative.head(8).itertuples(index=False):
            lines.append(
                f"- {row.ticker} {row.name}：{row.setup_type}，因子分 {row.factor_score:.1f}，"
                f"可操作分 {row.actionability_score:.1f}，20日相對 {row.relative_return_20d:.1%}，量比 {row.volume_ratio_20d:.2f}。"
            )

    lines.extend(
        [
            "",
            "## 可監控的早期訊號",
            "- 產業先轉強：sector_top_quartile_20d + 個股 relative_strength_20d。",
            "- 量價突破：new_20d_high + volume_expansion + 個股相對 SPY 轉強。",
            "- 趨勢中回檔：ma50 > ma200、ma50 上彎、股價回到 MA20 附近、RSI 45-70。",
            "- 安靜整理：靠近 60 日高點，但 5 日量縮到 20 日均量以下。",
            "- 深跌反轉：大回撤 + 放量，勝率通常比趨勢型差，必須縮小倉位。",
        ]
    )
    return "\n".join(lines) + "\n"


def human_factor(name: str) -> str:
    labels = {
        "above_ma50": "站上 50 日線",
        "above_ma200": "站上 200 日線",
        "ma20_gt_ma50": "20 日線高於 50 日線",
        "ma50_gt_ma200": "50 日線高於 200 日線",
        "positive_ma20_slope": "20 日線上彎",
        "positive_ma50_slope": "50 日線上彎",
        "relative_strength_20d": "20 日相對 SPY 強勢",
        "relative_strength_60d": "60 日相對 SPY 強勢",
        "near_60d_high": "接近 60 日高點",
        "near_252d_high": "接近 252 日高點",
        "new_20d_high": "突破 20 日新高",
        "healthy_pullback_to_ma20": "上升趨勢回測 MA20",
        "volume_expansion": "當日放量",
        "volume_dryup": "短線量縮",
        "rsi_constructive": "RSI 45-70",
        "rsi_hot": "RSI 過熱",
        "low_atr": "低波動",
        "high_atr": "高波動",
        "sector_top_quartile_20d": "產業 20 日強度前 25%",
        "sector_top_quartile_60d": "產業 60 日強度前 25%",
        "sector_rotation_plus_stock_rs": "產業輪動 + 個股相對強",
        "breakout_with_volume": "放量突破",
        "quiet_base_near_high": "高位安靜整理",
        "uptrend_pullback": "上升趨勢回檔型",
        "deep_reversal_risk": "深跌反轉/軋空型",
    }
    return labels.get(name, name)

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

from .config import resolve_path
from .data import fetch_price_history_cached, load_holdings, load_universe, unique_tickers
from .indicators import compute_price_metrics, is_finite


@dataclass
class ScanResult:
    as_of: date
    market_state: dict[str, Any]
    portfolio: dict[str, Any]
    market_dashboard: pd.DataFrame
    sector_scores: pd.DataFrame
    candidates: pd.DataFrame
    holdings_review: pd.DataFrame
    warnings: list[str]


def run_scan(config: dict[str, Any], config_dir: Path, as_of: date) -> ScanResult:
    universe_path = resolve_path(config.get("universe_path", "data/universe_us_swing.csv"), config_dir)
    holdings_path = resolve_path(config.get("holdings_path", "data/holdings.csv"), config_dir)
    universe = load_universe(universe_path)
    holdings = load_holdings(holdings_path)

    dashboard_tickers = list(dict(config.get("market_dashboard", {})).keys())
    holding_tickers = holdings.loc[~holdings.get("is_cash", False), "ticker"].tolist() if not holdings.empty else []
    all_tickers = unique_tickers([*dashboard_tickers, *universe["ticker"].tolist(), *holding_tickers])

    data_fetch_cfg = config.get("data_fetch", {})
    cache_cfg = data_fetch_cfg.get("cache", {})
    price_data = fetch_price_history_cached(
        all_tickers,
        lookback_days=int(config.get("lookback_days", 320)),
        as_of=as_of,
        cache_dir=resolve_path(cache_cfg.get("dir", "data/price_cache"), config_dir),
        cache_enabled=bool(cache_cfg.get("enabled", True)),
        force_refresh=bool(cache_cfg.get("force_refresh", False)),
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

    market_dashboard = build_market_dashboard(config, price_data)
    candidates = build_candidate_scores(config, universe, price_data, market_dashboard)
    market_state = classify_market_state(config, market_dashboard, candidates)
    candidates = apply_candidate_categories(config, candidates, market_state)
    sector_scores = build_sector_scores(candidates)
    holdings_review, portfolio = build_holdings_review(config, holdings, universe, candidates, price_data, market_state)

    warnings: list[str] = []
    missing_prices = sorted(set(all_tickers) - set(price_data))
    if missing_prices:
        warnings.append(f"Missing price data for {len(missing_prices)} symbols: {', '.join(missing_prices[:20])}")
    if holdings.empty:
        warnings.append(f"No holdings file found or holdings empty: {holdings_path}")
    elif not bool(holdings.get("is_cash", pd.Series(dtype=bool)).any()):
        warnings.append("No CASH row found in holdings; cash allocation cannot be measured.")
    elif portfolio.get("cash_value", 0.0) < 0:
        warnings.append("Cash balance is negative after margin balance; apply controlled margin policy before any new buy.")
    elif portfolio.get("cash_value", 0.0) == 0:
        warnings.append("CASH row exists but cash value is zero; fill shares with available USD cash.")

    return ScanResult(
        as_of=as_of,
        market_state=market_state,
        portfolio=portfolio,
        market_dashboard=market_dashboard,
        sector_scores=sector_scores,
        candidates=candidates,
        holdings_review=holdings_review,
        warnings=warnings,
    )


def build_market_dashboard(config: dict[str, Any], price_data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    names = dict(config.get("market_dashboard", {}))
    for symbol, name in names.items():
        metrics = compute_price_metrics(price_data.get(str(symbol).upper(), pd.DataFrame()))
        rows.append({"symbol": str(symbol).upper(), "name": name, **metrics})
    return pd.DataFrame(rows)


def build_candidate_scores(
    config: dict[str, Any],
    universe: pd.DataFrame,
    price_data: dict[str, pd.DataFrame],
    market_dashboard: pd.DataFrame,
) -> pd.DataFrame:
    benchmark = str(config.get("market", {}).get("benchmark", "SPY")).upper()
    benchmark_row = first_row(market_dashboard[market_dashboard["symbol"].eq(benchmark)]) if not market_dashboard.empty else {}
    benchmark_return_20d = as_float(benchmark_row.get("return_20d"), 0.0)
    theme_scores = dict(config.get("structural_theme_scores", {}))
    scanner_cfg = config.get("scanner", {})
    high_risk_sectors = set(config.get("high_risk_sectors", []))
    min_avg_dollar_volume = float(scanner_cfg.get("min_avg_dollar_volume", 25_000_000))
    rows: list[dict[str, Any]] = []

    for item in universe.itertuples(index=False):
        ticker = str(item.ticker).upper()
        metrics = compute_price_metrics(price_data.get(ticker, pd.DataFrame()))
        close = as_float(metrics.get("close"))
        sector = str(item.sector)
        theme_score = float(theme_scores.get(sector, 0.50))
        fundamental_score = parse_score(getattr(item, "fundamental_score", math.nan), default=8.0, maximum=20.0)
        relative_20d = as_float(metrics.get("return_20d")) - benchmark_return_20d
        high_risk = is_high_risk(item, sector, metrics, high_risk_sectors)
        overextended = is_overextended(config, metrics)
        enough_liquidity = as_float(metrics.get("avg_dollar_volume_20d"), 0.0) >= min_avg_dollar_volume

        structure_score = clamp(25.0 * theme_score + (2.0 if relative_20d > 0.05 else 0.0), 0.0, 25.0)
        freshness_score = score_freshness(metrics, overextended)
        catalyst_score = score_catalyst(metrics, relative_20d)
        institutional_score = score_institutional_proxy(metrics, relative_20d, enough_liquidity)
        valuation_risk_score = score_valuation_risk(metrics, high_risk, overextended, enough_liquidity)
        total_score = (
            structure_score
            + fundamental_score
            + freshness_score
            + catalyst_score
            + institutional_score
            + valuation_risk_score
        )

        rows.append(
            {
                "ticker": ticker,
                "name": str(item.name),
                "sector": sector,
                "theme": getattr(item, "theme", ""),
                "risk_bucket": getattr(item, "risk_bucket", ""),
                "notes": getattr(item, "notes", ""),
                **metrics,
                "relative_return_20d": relative_20d,
                "structure_score_25": round(structure_score, 2),
                "fundamental_score_20": round(fundamental_score, 2),
                "freshness_score_20": round(freshness_score, 2),
                "catalyst_proxy_score_15": round(catalyst_score, 2),
                "institutional_proxy_score_10": round(institutional_score, 2),
                "valuation_risk_score_10": round(valuation_risk_score, 2),
                "total_score_100": round(total_score, 2),
                "high_risk": high_risk,
                "overextended": overextended,
                "liquidity_ok": enough_liquidity,
                "data_quality": data_quality_note(metrics, fundamental_score),
                "category": "",
                "action_bias": "",
                "first_tranche_pct": 0.0,
                "entry_low": math.nan,
                "entry_high": math.nan,
                "stop_loss": math.nan,
                "take_profit_1": math.nan,
                "no_chase_above": math.nan,
                "reason": "",
            }
        )

    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    return frame.sort_values(["total_score_100", "avg_dollar_volume_20d"], ascending=[False, False]).reset_index(drop=True)


def classify_market_state(
    config: dict[str, Any],
    market_dashboard: pd.DataFrame,
    candidates: pd.DataFrame,
) -> dict[str, Any]:
    risk_cfg = config.get("risk", {})
    dashboard = {row.symbol: row._asdict() for row in market_dashboard.itertuples(index=False)} if not market_dashboard.empty else {}
    spy = dashboard.get("SPY", {})
    qqq = dashboard.get("QQQ", {})
    smh = dashboard.get("SMH", {})
    iwm = dashboard.get("IWM", {})
    vix = dashboard.get("^VIX", {})

    breadth = compute_breadth(candidates)
    vix_close = as_float(vix.get("close"))
    red_vix = float(risk_cfg.get("red_vix", 25.0))
    yellow_vix = float(risk_cfg.get("yellow_vix", 20.0))
    spy_below_200 = as_float(spy.get("pct_from_ma200")) < 0
    qqq_below_200 = as_float(qqq.get("pct_from_ma200")) < 0
    qqq_below_50 = as_float(qqq.get("pct_from_ma50")) < 0
    smh_below_50 = as_float(smh.get("pct_from_ma50")) < 0
    iwm_below_50 = as_float(iwm.get("pct_from_ma50")) < 0

    if (is_finite(vix_close) and vix_close >= red_vix) or (spy_below_200 and qqq_below_200):
        risk_light = "red"
        day_type = "應減碼日"
        cash_target = "80%+"
        new_exposure_limit = 0.0
    elif (qqq_below_50 and smh_below_50 and iwm_below_50) or breadth["above_ma50_ratio"] < 0.35:
        risk_light = "orange"
        day_type = "防守日"
        cash_target = "60%-80%"
        new_exposure_limit = 0.03
    elif (is_finite(vix_close) and vix_close >= yellow_vix) or qqq_below_50 or smh_below_50 or breadth["adv_decl_ratio"] < 0.8:
        risk_light = "yellow"
        day_type = "觀望日"
        cash_target = "50%-70%"
        new_exposure_limit = 0.05
    elif (
        as_float(spy.get("pct_from_ma50")) > 0
        and as_float(qqq.get("pct_from_ma50")) > 0
        and as_float(smh.get("pct_from_ma50")) > 0
        and breadth["adv_decl_ratio"] >= 1.2
        and breadth["above_ma50_ratio"] >= 0.55
    ):
        risk_light = "green"
        day_type = "進攻日"
        cash_target = "10%-20%"
        new_exposure_limit = 0.18
    else:
        risk_light = "neutral"
        day_type = "可小買日"
        cash_target = "30%-50%"
        new_exposure_limit = 0.10

    margin_cfg = dict(risk_cfg.get("margin", {}))
    margin_enabled = bool(margin_cfg.get("enabled", True))
    margin_allowed_risk_lights = set(margin_cfg.get("allowed_risk_lights", ["green"]))
    margin_allowed_categories = list(margin_cfg.get("allowed_categories", ["A"]))
    margin_allowed = margin_enabled and risk_light in margin_allowed_risk_lights
    margin_policy = {
        "enabled": margin_enabled,
        "allowed": margin_allowed,
        "allowed_risk_lights": sorted(margin_allowed_risk_lights),
        "allowed_categories": margin_allowed_categories,
        "max_negative_cash_weight": float(margin_cfg.get("max_negative_cash_weight", 0.20)),
        "max_single_margin_trade_pct": float(margin_cfg.get("max_single_margin_trade_pct", 0.05)),
        "max_new_margin_exposure_pct": float(margin_cfg.get("max_new_margin_exposure_pct", 0.08)),
    }
    if not margin_enabled:
        margin_note = "策略設定未啟用融資。"
    elif not margin_allowed:
        margin_note = "目前風險燈號不允許新增融資。"
    else:
        margin_note = "受控融資只限 A 類、明確買點、非追高，且不得超過負現金上限。"

    return {
        "risk_light": risk_light,
        "market_day_type": day_type,
        "cash_target": cash_target,
        "new_exposure_limit": new_exposure_limit,
        "single_name_limit": float(risk_cfg.get("max_single_stock_weight", 0.12)),
        "sector_limit": float(risk_cfg.get("max_sector_weight", 0.30)),
        "margin_allowed": margin_allowed,
        "margin_policy": margin_policy,
        "margin_note": margin_note,
        **breadth,
    }


def apply_candidate_categories(
    config: dict[str, Any],
    candidates: pd.DataFrame,
    market_state: dict[str, Any],
) -> pd.DataFrame:
    if candidates.empty:
        return candidates

    risk_cfg = config.get("risk", {})
    result = candidates.copy()
    categories: list[str] = []
    action_biases: list[str] = []
    reasons: list[str] = []
    first_tranches: list[float] = []
    entry_lows: list[float] = []
    entry_highs: list[float] = []
    stop_losses: list[float] = []
    take_profits: list[float] = []
    no_chase_aboves: list[float] = []

    for row in result.itertuples(index=False):
        close = as_float(getattr(row, "close", math.nan))
        score = as_float(getattr(row, "total_score_100", math.nan), 0.0)
        high_risk = bool(getattr(row, "high_risk", False))
        overextended = bool(getattr(row, "overextended", False))
        liquidity_ok = bool(getattr(row, "liquidity_ok", False))
        category, action_bias, reason = categorize_candidate(row, market_state)
        first_tranche = 0.0
        if category in {"A", "E"} and is_finite(close):
            first_tranche = float(
                risk_cfg.get("high_beta_first_tranche_pct" if high_risk else "normal_first_tranche_pct", 0.10)
            )
        if category == "B" and score >= 85 and liquidity_ok and not overextended:
            first_tranche = min(0.10, float(risk_cfg.get("normal_first_tranche_pct", 0.25)) / 2)

        entry_low, entry_high = entry_zone(row, category)
        stop_loss = stop_loss_for(row, high_risk)
        take_profit = close * (1.15 if high_risk else 1.20) if is_finite(close) else math.nan
        no_chase_above = close * (1.02 if high_risk else 1.03) if is_finite(close) else math.nan

        categories.append(category)
        action_biases.append(action_bias)
        reasons.append(reason)
        first_tranches.append(round(first_tranche, 4))
        entry_lows.append(entry_low)
        entry_highs.append(entry_high)
        stop_losses.append(stop_loss)
        take_profits.append(take_profit)
        no_chase_aboves.append(no_chase_above)

    result["category"] = categories
    result["action_bias"] = action_biases
    result["reason"] = reasons
    result["first_tranche_pct"] = first_tranches
    result["entry_low"] = entry_lows
    result["entry_high"] = entry_highs
    result["stop_loss"] = stop_losses
    result["take_profit_1"] = take_profits
    result["no_chase_above"] = no_chase_aboves
    return result.sort_values(["category", "total_score_100"], ascending=[True, False]).reset_index(drop=True)


def categorize_candidate(row: Any, market_state: dict[str, Any]) -> tuple[str, str, str]:
    close = as_float(getattr(row, "close", math.nan))
    score = as_float(getattr(row, "total_score_100", math.nan), 0.0)
    day_type = str(market_state.get("market_day_type", "可小買日"))
    risk_light = str(market_state.get("risk_light", "neutral"))
    high_risk = bool(getattr(row, "high_risk", False))
    overextended = bool(getattr(row, "overextended", False))
    liquidity_ok = bool(getattr(row, "liquidity_ok", False))
    catalyst = as_float(getattr(row, "catalyst_proxy_score_15", math.nan), 0.0)
    pct_ma50 = as_float(getattr(row, "pct_from_ma50", math.nan))
    pct_ma200 = as_float(getattr(row, "pct_from_ma200", math.nan))

    if not is_finite(close):
        return "G", "保留現金", "價格資料不足，不能建立可執行價位。"
    if overextended:
        return "C", "不追", "已過度延伸，停損距離與追價風險偏高。"
    if risk_light == "red":
        return "G", "保留現金", "市場紅燈，優先降低風險。"
    if risk_light == "orange" and score < 85:
        return "G", "保留現金", "市場防守日，非高分標的不新增。"
    if high_risk and score >= 70:
        return "E", "極小倉觀察", "高 beta 或題材屬性，只能用小倉位驗證。"
    if score >= 80 and day_type in {"進攻日", "可小買日"} and liquidity_ok:
        return "A", "可試第一筆", "分數達標且市場允許，但仍需分批。"
    if score >= 75 and pct_ma50 > 0 and pct_ma200 > 0 and catalyst < 7:
        return "D", "中期抱/等催化", "趨勢健康但近期催化不足，適合等更清楚觸發。"
    if score >= 70:
        return "B", "等回檔", "主線或技術條件尚可，但買點不夠低風險。"
    return "G", "保留現金", "分數不足，沒有比現金更好的交易。"


def build_sector_scores(candidates: pd.DataFrame) -> pd.DataFrame:
    if candidates.empty:
        return pd.DataFrame()
    grouped = candidates.groupby("sector", dropna=False)
    rows = []
    for sector, frame in grouped:
        top_mover = top_abs_mover(frame)
        rows.append(
            {
                "sector": sector,
                "count": len(frame),
                "avg_score": frame["total_score_100"].mean(),
                "avg_return_1d": frame["return_1d"].mean(),
                "avg_abs_return_1d": frame["return_1d"].abs().mean(),
                "avg_return_5d": frame["return_5d"].mean(),
                "avg_return_20d": frame["return_20d"].mean(),
                "top_mover": top_mover.get("ticker", ""),
                "top_mover_return_1d": top_mover.get("return_1d", math.nan),
                "advancers": int(frame["return_1d"].gt(0).sum()),
                "decliners": int(frame["return_1d"].lt(0).sum()),
                "category_a_count": int(frame["category"].eq("A").sum()),
                "category_b_count": int(frame["category"].eq("B").sum()),
                "overextended_count": int(frame["overextended"].sum()),
            }
        )
    return pd.DataFrame(rows).sort_values(["avg_score", "avg_return_20d"], ascending=[False, False]).reset_index(drop=True)


def build_holdings_review(
    config: dict[str, Any],
    holdings: pd.DataFrame,
    universe: pd.DataFrame,
    candidates: pd.DataFrame,
    price_data: dict[str, pd.DataFrame],
    market_state: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if holdings.empty:
        return pd.DataFrame(), {
            "cash_value": 0.0,
            "stock_value": 0.0,
            "total_equity": 0.0,
            "cash_weight": math.nan,
            "sector_exposure": [],
            "note": "No holdings file. Portfolio-first decisions are limited.",
        }

    universe_lookup = universe.set_index("ticker").to_dict("index") if not universe.empty else {}
    candidate_lookup = candidates.set_index("ticker").to_dict("index") if not candidates.empty else {}
    cash_rows = holdings[holdings["is_cash"]].copy()
    cash_value = 0.0
    if not cash_rows.empty:
        for row in cash_rows.itertuples(index=False):
            cash_value += float(row.shares) if float(row.shares) != 0 else float(row.avg_cost)

    rows: list[dict[str, Any]] = []
    stock_value = 0.0
    for row in holdings[~holdings["is_cash"]].itertuples(index=False):
        ticker = str(row.ticker).upper()
        metrics = compute_price_metrics(price_data.get(ticker, pd.DataFrame()))
        close = as_float(metrics.get("close"))
        shares = float(row.shares)
        avg_cost = float(row.avg_cost)
        market_value = shares * close if is_finite(close) else 0.0
        stock_value += market_value
        cost_value = shares * avg_cost
        pnl_pct = (market_value / cost_value - 1) if cost_value > 0 and market_value > 0 else math.nan
        meta = universe_lookup.get(ticker, fallback_holding_meta(ticker))
        candidate = candidate_lookup.get(ticker, {})
        sector = str(meta.get("sector", candidate.get("sector", "")))
        high_risk = bool(candidate.get("high_risk", is_high_risk_meta(meta)))
        action, sell_pct, action_reason = holding_action(row, metrics, pnl_pct, high_risk, market_state, candidate)
        stop_loss = holding_stop_loss(metrics, avg_cost, pnl_pct)
        take_profit = close * (1.15 if high_risk else 1.20) if is_finite(close) else math.nan
        rows.append(
            {
                "ticker": ticker,
                "name": row.name or meta.get("name", ticker),
                "sector": sector,
                "shares": shares,
                "avg_cost": avg_cost,
                "close": close,
                "market_value": market_value,
                "unrealized_pnl_pct": pnl_pct,
                "portfolio_weight": math.nan,
                "trade_type": row.trade_type,
                "thesis": row.thesis,
                "model_score": candidate.get("total_score_100", math.nan),
                "model_category": candidate.get("category", ""),
                "action": action,
                "suggested_sell_pct": sell_pct,
                "stop_loss": stop_loss,
                "take_profit_ref": take_profit,
                "reason": action_reason,
                **{f"price_{key}": value for key, value in metrics.items() if key in {"return_5d", "return_20d", "pct_from_ma20", "pct_from_ma50", "atr_pct"}},
            }
        )

    total_equity = cash_value + stock_value
    review = pd.DataFrame(rows)
    if not review.empty and total_equity > 0:
        review["portfolio_weight"] = review["market_value"] / total_equity

    sector_exposure = []
    if not review.empty and total_equity > 0:
        sector_exposure = (
            review.groupby("sector")["market_value"]
            .sum()
            .sort_values(ascending=False)
            .reset_index()
            .assign(weight=lambda frame: frame["market_value"] / total_equity)
            .to_dict("records")
        )

    portfolio = {
        "cash_value": cash_value,
        "stock_value": stock_value,
        "total_equity": total_equity,
        "cash_weight": cash_value / total_equity if total_equity > 0 else math.nan,
        "sector_exposure": sector_exposure,
        "max_single_stock_weight": float(config.get("risk", {}).get("max_single_stock_weight", 0.12)),
        "max_sector_weight": float(config.get("risk", {}).get("max_sector_weight", 0.30)),
    }
    return review, portfolio


def scan_to_context(
    result: ScanResult,
    top_candidates: int = 40,
    after_close_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    candidates = result.candidates.head(int(top_candidates)).copy() if not result.candidates.empty else pd.DataFrame()
    return {
        "as_of": result.as_of.isoformat(),
        "market_state": result.market_state,
        "portfolio": result.portfolio,
        "warnings": result.warnings,
        "market_dashboard": records(result.market_dashboard),
        "sector_scores": records(result.sector_scores.head(20)),
        "holdings_review": records(result.holdings_review),
        "candidate_summary": {
            "configured_universe_candidates": int(len(result.candidates)),
            "category_counts": result.candidates["category"].value_counts().to_dict() if not result.candidates.empty else {},
            "note": "This is the configured project universe, not a literal scan of every US listed stock.",
        },
        "top_candidates": records(candidates),
        "after_close_review": build_after_close_review(result, after_close_config or {}),
    }


def build_after_close_review(result: ScanResult, config: dict[str, Any]) -> dict[str, Any]:
    candidates = result.candidates.copy()
    holdings = result.holdings_review.copy()
    sectors = result.sector_scores.copy()
    if candidates.empty:
        return {
            "note": "No candidate data available for after-close review.",
            "news_watch_symbols": [],
        }

    max_rows = int(config.get("max_rows_per_section", 8))
    big_move_1d_abs = float(config.get("big_move_1d_abs", 0.04))
    big_move_5d_abs = float(config.get("big_move_5d_abs", 0.08))
    missed_buy_return_1d = float(config.get("missed_buy_return_1d", 0.015))
    missed_buy_return_5d = float(config.get("missed_buy_return_5d", 0.035))
    min_candidate_score = float(config.get("min_candidate_score", 75.0))
    holding_tickers = set(holdings["ticker"].astype(str).str.upper()) if not holdings.empty and "ticker" in holdings else set()

    review_columns = [
        "ticker",
        "name",
        "sector",
        "category",
        "action_bias",
        "total_score_100",
        "close",
        "return_1d",
        "return_5d",
        "return_20d",
        "volume_ratio_20d",
        "entry_low",
        "entry_high",
        "stop_loss",
        "no_chase_above",
        "reason",
    ]
    holding_columns = [
        "ticker",
        "name",
        "sector",
        "shares",
        "close",
        "market_value",
        "portfolio_weight",
        "unrealized_pnl_pct",
        "model_score",
        "model_category",
        "action",
        "suggested_sell_pct",
        "stop_loss",
        "take_profit_ref",
        "reason",
    ]

    unheld = candidates[~candidates["ticker"].astype(str).str.upper().isin(holding_tickers)].copy()
    candidate_buy = unheld[
        (
            unheld["category"].isin(["A", "E"])
            | ((unheld["category"].eq("B")) & (pd.to_numeric(unheld["total_score_100"], errors="coerce") >= min_candidate_score))
        )
        & (
            (pd.to_numeric(unheld["return_1d"], errors="coerce") >= missed_buy_return_1d)
            | (pd.to_numeric(unheld["return_5d"], errors="coerce") >= missed_buy_return_5d)
        )
    ].copy()
    candidate_add = pd.DataFrame()
    if not holdings.empty:
        add_mask = holdings["action"].astype(str).str.contains("加碼", na=False)
        add_mask &= pd.to_numeric(holdings.get("suggested_sell_pct", 0), errors="coerce").fillna(0).le(0)
        candidate_add = holdings[add_mask].copy()

    candidate_sell = pd.DataFrame()
    if not holdings.empty:
        sell_mask = pd.to_numeric(holdings.get("suggested_sell_pct", 0), errors="coerce").fillna(0).gt(0)
        sell_mask |= holdings["action"].astype(str).str.contains("減碼|停損|停利", regex=True, na=False)
        candidate_sell = holdings[sell_mask].copy()

    big_gainers = candidates[pd.to_numeric(candidates["return_1d"], errors="coerce") >= big_move_1d_abs].copy()
    big_losers = candidates[pd.to_numeric(candidates["return_1d"], errors="coerce") <= -big_move_1d_abs].copy()
    large_abs = candidates.assign(abs_return_1d=pd.to_numeric(candidates["return_1d"], errors="coerce").abs())
    top_5d = candidates[pd.to_numeric(candidates["return_5d"], errors="coerce") >= big_move_5d_abs].copy()
    weak_5d = candidates[pd.to_numeric(candidates["return_5d"], errors="coerce") <= -big_move_5d_abs].copy()
    new_highs = candidates[pd.to_numeric(candidates["drawdown_from_20d_high"], errors="coerce") >= -0.01].copy()
    near_lows = candidates[pd.to_numeric(candidates["pct_above_20d_low"], errors="coerce") <= 0.02].copy()
    no_chase = candidates[candidates["category"].eq("C")].copy()

    review = {
        "scope_note": "Review is based on configured universe plus current holdings, not every US-listed stock.",
        "missed_buy_candidates": frame_records(candidate_buy, review_columns, max_rows, "total_score_100"),
        "missed_add_candidates": frame_records(candidate_add, holding_columns, max_rows, "model_score"),
        "missed_sell_candidates": frame_records(candidate_sell, holding_columns, max_rows, "suggested_sell_pct"),
        "big_gainers_1d": frame_records(big_gainers, review_columns, max_rows, "return_1d"),
        "big_losers_1d": frame_records(big_losers, review_columns, max_rows, "return_1d", ascending=True),
        "largest_abs_moves_1d": frame_records(large_abs, review_columns + ["abs_return_1d"], max_rows, "abs_return_1d"),
        "top_5d_momentum": frame_records(top_5d, review_columns, max_rows, "return_5d"),
        "weak_5d_momentum": frame_records(weak_5d, review_columns, max_rows, "return_5d", ascending=True),
        "new_20d_highs": frame_records(new_highs, review_columns, max_rows, "return_20d"),
        "near_20d_lows": frame_records(near_lows, review_columns, max_rows, "return_20d", ascending=True),
        "overextended_no_chase": frame_records(no_chase, review_columns, max_rows, "return_20d"),
        "sector_movers_up": frame_records(sectors, None, max_rows, "avg_return_1d"),
        "sector_movers_down": frame_records(sectors, None, max_rows, "avg_return_1d", ascending=True),
        "portfolio_gaps": portfolio_gap_review(result.portfolio, holdings),
    }
    review["news_watch_symbols"] = news_watch_symbols(review)
    return review


def top_abs_mover(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty or "return_1d" not in frame:
        return {}
    returns = pd.to_numeric(frame["return_1d"], errors="coerce").abs()
    if returns.dropna().empty:
        return {}
    row = frame.loc[returns.idxmax()]
    return row.to_dict()


def frame_records(
    frame: pd.DataFrame,
    columns: list[str] | None,
    limit: int,
    sort_by: str,
    ascending: bool = False,
) -> list[dict[str, Any]]:
    if frame is None or frame.empty:
        return []
    output = frame.copy()
    if sort_by in output:
        output[sort_by] = pd.to_numeric(output[sort_by], errors="coerce")
        output = output.sort_values(sort_by, ascending=ascending, na_position="last")
    if columns is not None:
        selected = [column for column in columns if column in output]
        output = output.loc[:, selected]
    return records(output.head(max(1, int(limit))))


def portfolio_gap_review(portfolio: dict[str, Any], holdings: pd.DataFrame) -> list[dict[str, Any]]:
    gaps: list[dict[str, Any]] = []
    cash_value = as_float(portfolio.get("cash_value"), 0.0)
    if cash_value < 0:
        gaps.append(
            {
                "type": "negative_cash",
                "severity": "high",
                "message": "現金為負，新增買進應視為融資；盤後優先檢討是否要降槓桿。",
                "value": cash_value,
            }
        )
    max_single = as_float(portfolio.get("max_single_stock_weight"), 0.12)
    if not holdings.empty and "portfolio_weight" in holdings:
        overweight = holdings[pd.to_numeric(holdings["portfolio_weight"], errors="coerce") > max_single].copy()
        for row in overweight.sort_values("portfolio_weight", ascending=False).head(5).itertuples(index=False):
            gaps.append(
                {
                    "type": "single_name_overweight",
                    "severity": "medium",
                    "ticker": getattr(row, "ticker", ""),
                    "message": "單檔權重高於策略上限，若技術面轉弱要優先降。",
                    "value": as_float(getattr(row, "portfolio_weight", math.nan)),
                }
            )
    max_sector = as_float(portfolio.get("max_sector_weight"), 0.30)
    for item in portfolio.get("sector_exposure", []) or []:
        weight = as_float(item.get("weight"))
        if weight > max_sector:
            gaps.append(
                {
                    "type": "sector_overweight",
                    "severity": "medium",
                    "sector": item.get("sector", ""),
                    "message": "產業權重高於策略上限，隔日買進需避開同產業。",
                    "value": weight,
                }
            )
    return gaps


def news_watch_symbols(review: dict[str, Any]) -> list[str]:
    symbols: list[str] = []
    for key in [
        "missed_sell_candidates",
        "missed_add_candidates",
        "missed_buy_candidates",
        "big_gainers_1d",
        "big_losers_1d",
        "largest_abs_moves_1d",
    ]:
        for row in review.get(key, []) or []:
            ticker = str(row.get("ticker", "")).strip().upper()
            if ticker and ticker not in symbols:
                symbols.append(ticker)
    return symbols[:20]


def compute_breadth(candidates: pd.DataFrame) -> dict[str, Any]:
    if candidates.empty:
        return {
            "advancers": 0,
            "decliners": 0,
            "adv_decl_ratio": 0.0,
            "above_ma50_ratio": 0.0,
            "new_20d_high_count": 0,
            "new_20d_low_count": 0,
        }
    advancers = int(candidates["return_1d"].gt(0).sum())
    decliners = int(candidates["return_1d"].lt(0).sum())
    ratio = advancers / decliners if decliners else float(advancers)
    above_ma50 = candidates["pct_from_ma50"].gt(0)
    new_highs = candidates["drawdown_from_20d_high"].ge(-0.01)
    near_lows = candidates["pct_above_20d_low"].le(0.01)
    return {
        "advancers": advancers,
        "decliners": decliners,
        "adv_decl_ratio": round(ratio, 3),
        "above_ma50_ratio": round(float(above_ma50.mean()), 3),
        "new_20d_high_count": int(new_highs.sum()),
        "new_20d_low_count": int(near_lows.sum()),
    }


def score_freshness(metrics: dict[str, Any], overextended: bool) -> float:
    if not is_finite(metrics.get("close")):
        return 0.0
    score = 6.0
    ret20 = as_float(metrics.get("return_20d"))
    pct_ma20 = as_float(metrics.get("pct_from_ma20"))
    pct_ma50 = as_float(metrics.get("pct_from_ma50"))
    pct_ma200 = as_float(metrics.get("pct_from_ma200"))
    drawdown = as_float(metrics.get("drawdown_from_20d_high"))
    if pct_ma50 > 0:
        score += 4.0
    if pct_ma200 > 0:
        score += 3.0
    if -0.05 <= ret20 <= 0.18:
        score += 3.0
    if -0.06 <= pct_ma20 <= 0.08:
        score += 2.0
    if -0.10 <= drawdown <= -0.01:
        score += 2.0
    if overextended:
        score -= 5.0
    return clamp(score, 0.0, 20.0)


def score_catalyst(metrics: dict[str, Any], relative_20d: float) -> float:
    if not is_finite(metrics.get("close")):
        return 0.0
    score = 3.0
    if as_float(metrics.get("return_5d")) > 0:
        score += 2.0
    if as_float(metrics.get("return_20d")) > 0:
        score += 2.0
    if relative_20d > 0.03:
        score += 2.0
    if as_float(metrics.get("volume_ratio_20d")) >= 1.2:
        score += 2.0
    if as_float(metrics.get("drawdown_from_20d_high")) >= -0.04:
        score += 2.0
    return clamp(score, 0.0, 15.0)


def score_institutional_proxy(metrics: dict[str, Any], relative_20d: float, liquidity_ok: bool) -> float:
    if not is_finite(metrics.get("close")):
        return 0.0
    score = 2.0
    if liquidity_ok:
        score += 3.0
    if relative_20d > 0:
        score += 2.0
    if as_float(metrics.get("volume_ratio_20d")) >= 1.1 and as_float(metrics.get("return_5d")) > 0:
        score += 2.0
    if as_float(metrics.get("avg_dollar_volume_20d"), 0.0) >= 250_000_000:
        score += 1.0
    return clamp(score, 0.0, 10.0)


def score_valuation_risk(metrics: dict[str, Any], high_risk: bool, overextended: bool, liquidity_ok: bool) -> float:
    if not is_finite(metrics.get("close")):
        return 0.0
    score = 8.0
    atr_pct = as_float(metrics.get("atr_pct"))
    pct_ma20 = as_float(metrics.get("pct_from_ma20"))
    if high_risk:
        score -= 1.5
    if overextended:
        score -= 3.0
    if atr_pct > 0.06:
        score -= 2.0
    elif atr_pct < 0.035:
        score += 1.0
    if pct_ma20 > 0.12:
        score -= 1.0
    if not liquidity_ok:
        score -= 2.0
    return clamp(score, 0.0, 10.0)


def is_overextended(config: dict[str, Any], metrics: dict[str, Any]) -> bool:
    scanner_cfg = config.get("scanner", {})
    if not is_finite(metrics.get("close")):
        return False
    return (
        as_float(metrics.get("return_1d")) >= float(scanner_cfg.get("overextended_return_1d", 0.10))
        or as_float(metrics.get("return_20d")) >= float(scanner_cfg.get("overextended_return_20d", 0.35))
        or as_float(metrics.get("pct_from_ma20")) >= float(scanner_cfg.get("overextended_pct_above_ma20", 0.15))
    )


def is_high_risk(item: Any, sector: str, metrics: dict[str, Any], high_risk_sectors: set[str]) -> bool:
    risk_bucket = str(getattr(item, "risk_bucket", "")).lower()
    ticker = str(getattr(item, "ticker", "")).upper()
    atr_pct = as_float(metrics.get("atr_pct"))
    return (
        "high" in risk_bucket
        or "space" in risk_bucket
        or ticker in {"RKLB", "LUNR", "ASTS", "PL", "RDW", "BKSY", "SPIR", "DXYZ"}
        or (sector in high_risk_sectors and atr_pct >= 0.045)
    )


def is_high_risk_meta(meta: dict[str, Any]) -> bool:
    risk_bucket = str(meta.get("risk_bucket", "")).lower()
    sector = str(meta.get("sector", ""))
    return "high" in risk_bucket or "space" in risk_bucket or "space" in sector.lower()


def fallback_holding_meta(ticker: str) -> dict[str, Any]:
    overrides = {
        "DAL": {"sector": "Airlines and travel", "risk_bucket": "cyclical"},
        "IVV": {"sector": "Broad market ETF", "risk_bucket": "etf"},
        "SMH": {"sector": "Semiconductor ETF", "risk_bucket": "etf"},
    }
    return overrides.get(str(ticker).upper(), {})


def data_quality_note(metrics: dict[str, Any], fundamental_score: float) -> str:
    notes = []
    if not is_finite(metrics.get("close")):
        notes.append("missing_price")
    if fundamental_score <= 8.0:
        notes.append("fundamentals_not_verified")
    if as_float(metrics.get("avg_dollar_volume_20d"), 0.0) <= 0:
        notes.append("liquidity_unknown")
    return ",".join(notes) if notes else "ok"


def entry_zone(row: Any, category: str) -> tuple[float, float]:
    close = as_float(getattr(row, "close", math.nan))
    ma20 = as_float(getattr(row, "ma20", math.nan))
    ma50 = as_float(getattr(row, "ma50", math.nan))
    if not is_finite(close):
        return math.nan, math.nan
    if category == "A":
        return close * 0.985, close * 1.010
    if category == "E":
        return close * 0.970, close
    if category in {"B", "D", "C"}:
        anchor = ma20 if is_finite(ma20) else ma50 if is_finite(ma50) else close * 0.95
        return anchor * 0.985, anchor * 1.010
    return math.nan, math.nan


def stop_loss_for(row: Any, high_risk: bool) -> float:
    close = as_float(getattr(row, "close", math.nan))
    ma50 = as_float(getattr(row, "ma50", math.nan))
    atr = as_float(getattr(row, "atr14", math.nan))
    if not is_finite(close):
        return math.nan
    base = close * (0.90 if high_risk else 0.935)
    if is_finite(ma50) and close > ma50:
        base = max(base, ma50 * 0.97)
    if is_finite(atr):
        base = max(base, close - 2.2 * atr)
    return min(base, close * 0.985)


def holding_action(
    row: Any,
    metrics: dict[str, Any],
    pnl_pct: float,
    high_risk: bool,
    market_state: dict[str, Any],
    candidate: dict[str, Any],
) -> tuple[str, float, str]:
    close = as_float(metrics.get("close"))
    if not is_finite(close):
        return "等待", 0.0, "價格資料不足，不能調整。"
    risk_light = str(market_state.get("risk_light", "neutral"))
    pct_ma50 = as_float(metrics.get("pct_from_ma50"))
    if is_finite(pnl_pct) and pnl_pct <= -0.10:
        return "停損/減碼", 0.50, "虧損已達 -10% 附近，不能把波段凹成長投。"
    if pct_ma50 < 0 and (not is_finite(pnl_pct) or pnl_pct < 0.05):
        return "減碼", 0.25, "跌破 50 日線且獲利墊不足。"
    if risk_light in {"red", "orange"} and high_risk:
        return "減碼", 0.25, "市場風險升級，高 beta 先降曝險。"
    if high_risk and is_finite(pnl_pct) and pnl_pct >= 0.40:
        return "停利到半倉", 0.50, "高 beta 獲利超過 40%，至少鎖一部分。"
    if high_risk and is_finite(pnl_pct) and pnl_pct >= 0.15:
        return "停利 20%", 0.20, "高 beta 已達第一段停利區。"
    if is_finite(pnl_pct) and pnl_pct >= 0.20:
        return "移動停利", 0.0, "已有明顯獲利，優先用移動停利保護。"
    if candidate.get("category") == "A" and risk_light in {"green", "neutral"}:
        return "抱/可小幅加碼", 0.0, "模型仍列 A 類，若回測不破可小幅加碼。"
    return "抱/等待", 0.0, "沒有足夠理由主動交易。"


def holding_stop_loss(metrics: dict[str, Any], avg_cost: float, pnl_pct: float) -> float:
    close = as_float(metrics.get("close"))
    if not is_finite(close):
        return math.nan
    ma50 = as_float(metrics.get("ma50"))
    atr = as_float(metrics.get("atr14"))
    if is_finite(pnl_pct) and pnl_pct > 0.10:
        candidates = [close * 0.92]
        if is_finite(ma50):
            candidates.append(ma50 * 0.97)
        if is_finite(atr):
            candidates.append(close - 2.2 * atr)
        return min(max(candidates), close * 0.985)
    return min(avg_cost * 0.92 if avg_cost > 0 else close * 0.92, close * 0.985)


def parse_score(value: Any, default: float, maximum: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(parsed):
        return default
    return clamp(parsed, 0.0, maximum)


def first_row(frame: pd.DataFrame) -> dict[str, Any]:
    return frame.iloc[0].to_dict() if frame is not None and not frame.empty else {}


def as_float(value: Any, default: float = math.nan) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if math.isfinite(parsed) else default


def clamp(value: float, lower: float, upper: float) -> float:
    return min(max(float(value), lower), upper)


def records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    if frame is None or frame.empty:
        return []
    cleaned = frame.copy()
    cleaned = cleaned.where(pd.notna(cleaned), None)
    return cleaned.to_dict("records")

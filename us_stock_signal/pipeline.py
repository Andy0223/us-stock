from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

from .closed_loop import build_closed_loop_context, closed_loop_watchlist_rows, radar_to_frame
from .config import resolve_path
from .earnings import collect_earnings_calendar, rows_to_frame
from .market_calendar import is_us_early_close, is_us_trading_day, next_trading_day, ny_today
from .news import collect_symbol_news
from .notify import send_telegram_message
from .options import collect_options_review
from .scanner import ScanResult, run_scan, scan_to_context
from .strategy import StrategyReport, generate_strategy_report, sanitize_for_json
from .trades import load_trade_log
from .watchlist import build_next_day_watchlist, evaluate_watchlist, load_active_watchlist, write_watchlist


@dataclass
class PipelineResult:
    as_of: date
    mode: str
    report: StrategyReport
    scan: ScanResult | None
    context_path: Path | None
    report_path: Path | None
    market_dashboard_path: Path | None
    sector_scores_path: Path | None
    candidates_path: Path | None
    holdings_review_path: Path | None
    options_review_path: Path | None
    earnings_calendar_path: Path | None
    watchlist_path: Path | None
    closed_loop_radar_path: Path | None
    telegram_sent: bool
    telegram_error: str | None = None


def run_pipeline(
    config: dict[str, Any],
    config_dir: Path,
    mode: str | None = None,
    as_of: date | None = None,
    force: bool = False,
    skip_ai: bool = False,
    dry_run: bool = False,
) -> PipelineResult:
    run_mode = mode or str(config.get("mode", "premarket"))
    market_cfg = config.get("market", {})
    run_date = as_of or ny_today(str(market_cfg.get("timezone", "America/New_York")))
    output_dir = resolve_path(config.get("output_dir", "outputs"), config_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if bool(market_cfg.get("skip_non_trading_days", True)) and not force and not is_us_trading_day(run_date):
        text = f"美股波段策略｜{run_date.isoformat()}\n今日不是美股正常交易日，未執行掃描與交易建議。"
        report = StrategyReport(as_of=run_date, mode=run_mode, text=text, provider_status="non_trading_day")
        telegram_sent = False
        telegram_error = None
        if bool(config.get("telegram", {}).get("enabled", False)) and not dry_run:
            try:
                send_telegram_message(config["telegram"], text)
                telegram_sent = True
            except Exception as exc:  # pragma: no cover - network/provider dependent
                telegram_error = str(exc)
        return PipelineResult(
            as_of=run_date,
            mode=run_mode,
            report=report,
            scan=None,
            context_path=None,
            report_path=None,
            market_dashboard_path=None,
            sector_scores_path=None,
            candidates_path=None,
            holdings_review_path=None,
            options_review_path=None,
            earnings_calendar_path=None,
            watchlist_path=None,
            closed_loop_radar_path=None,
            telegram_sent=telegram_sent,
            telegram_error=telegram_error,
        )

    scan = run_scan(config, config_dir, run_date)
    context = scan_to_context(
        scan,
        int(config.get("scanner", {}).get("top_candidates", 40)),
        dict(config.get("after_close_review", {})),
    )
    context["mode"] = run_mode
    context["market_calendar"] = {
        "is_trading_day": is_us_trading_day(run_date),
        "is_early_close": is_us_early_close(run_date),
    }
    trade_cfg = dict(config.get("trade_log", {}))
    if bool(trade_cfg.get("enabled", True)):
        trade_path = resolve_path(trade_cfg.get("path", "data/trades.csv"), config_dir)
        context["trade_log"] = load_trade_log(trade_path, run_date)
    watchlist_cfg = dict(config.get("watchlist", {}))
    watchlist_path = resolve_path(watchlist_cfg.get("path", "data/watchlist_next_day.csv"), config_dir)
    active_watchlist = load_active_watchlist(watchlist_path, run_date)
    context["watchlist_review"] = evaluate_watchlist(active_watchlist, context, price_map=scan_price_lookup(scan))
    context["closed_loop"] = build_closed_loop_context(
        config,
        config_dir,
        context,
        run_date,
        run_mode,
        candidate_frame=scan.candidates,
        sector_score_frame=scan.sector_scores,
        market_dashboard_frame=scan.market_dashboard,
    )
    if bool(config.get("earnings_calendar", {}).get("enabled", True)):
        context["earnings_calendar"] = collect_earnings_calendar(
            earnings_watch_symbols(context),
            as_of=run_date,
            config=dict(config.get("earnings_calendar", {})),
        )
    if run_mode == "after_close" and bool(config.get("news_review", {}).get("enabled", True)):
        news_cfg = config.get("news_review", {})
        context["news_review"] = collect_symbol_news(
            context.get("after_close_review", {}).get("news_watch_symbols", []),
            max_symbols=int(news_cfg.get("max_symbols", 14)),
            max_items_per_symbol=int(news_cfg.get("max_items_per_symbol", 2)),
        )
    if bool(config.get("options_review", {}).get("enabled", True)):
        option_cfg = dict(config.get("options_review", {}))
        context["options_review"] = collect_options_review(
            options_review_symbols(context, option_cfg),
            price_lookup=price_lookup(context),
            as_of=run_date,
            config=option_cfg,
        )
    if run_mode == "after_close" and bool(watchlist_cfg.get("enabled", True)):
        valid_for = next_trading_day(run_date)
        watchlist_items = build_next_day_watchlist(context, valid_for)
        if bool(config.get("closed_loop", {}).get("watchlist_enabled", True)):
            watchlist_items.extend(
                closed_loop_watchlist_rows(
                    context,
                    valid_for,
                    max_rows=int(config.get("closed_loop", {}).get("watchlist_max_rows", 6)),
                )
            )
        context["watchlist_created"] = {
            "valid_for": valid_for.isoformat(),
            "items": watchlist_items,
        }
    report = generate_strategy_report(config, config_dir, context, run_mode, run_date, skip_ai=skip_ai)

    stamp = run_date.strftime("%Y%m%d")
    context_path = output_dir / f"us_swing_context_{run_mode}_{stamp}.json"
    report_path = output_dir / f"us_swing_report_{run_mode}_{stamp}.md"
    market_dashboard_path = output_dir / f"market_dashboard_{stamp}.csv"
    sector_scores_path = output_dir / f"sector_scores_{stamp}.csv"
    candidates_path = output_dir / f"candidate_scores_{stamp}.csv"
    holdings_review_path = output_dir / f"holdings_review_{stamp}.csv"
    options_review_path = output_dir / f"options_review_{stamp}.csv"
    earnings_calendar_path = output_dir / f"earnings_calendar_{stamp}.csv"
    closed_loop_radar_path = output_dir / f"closed_loop_radar_{stamp}.csv"

    context_path.write_text(json.dumps(sanitize_for_json(context), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report_path.write_text(report.text + "\n", encoding="utf-8")
    scan.market_dashboard.to_csv(market_dashboard_path, index=False)
    scan.sector_scores.to_csv(sector_scores_path, index=False)
    scan.candidates.to_csv(candidates_path, index=False)
    scan.holdings_review.to_csv(holdings_review_path, index=False)
    pd.DataFrame(context.get("options_review", {}).get("rows", [])).to_csv(options_review_path, index=False)
    rows_to_frame(context.get("earnings_calendar", {})).to_csv(earnings_calendar_path, index=False)
    radar_to_frame(context.get("closed_loop", {})).to_csv(closed_loop_radar_path, index=False)
    if run_mode == "after_close" and bool(watchlist_cfg.get("enabled", True)):
        write_watchlist(watchlist_path, context.get("watchlist_created", {}).get("items", []))

    telegram_sent = False
    telegram_error = None
    if bool(config.get("telegram", {}).get("enabled", False)) and not dry_run:
        try:
            send_telegram_message(config["telegram"], report.text)
            telegram_sent = True
        except Exception as exc:  # pragma: no cover - network/provider dependent
            telegram_error = str(exc)

    return PipelineResult(
        as_of=run_date,
        mode=run_mode,
        report=report,
        scan=scan,
        context_path=context_path,
        report_path=report_path,
        market_dashboard_path=market_dashboard_path,
        sector_scores_path=sector_scores_path,
        candidates_path=candidates_path,
        holdings_review_path=holdings_review_path,
        options_review_path=options_review_path,
        earnings_calendar_path=earnings_calendar_path,
        watchlist_path=watchlist_path,
        closed_loop_radar_path=closed_loop_radar_path,
        telegram_sent=telegram_sent,
        telegram_error=telegram_error,
    )


def options_review_symbols(context: dict[str, Any], option_cfg: dict[str, Any]) -> list[str]:
    symbols: list[str] = []
    top_candidate_limit = int(option_cfg.get("top_candidate_symbols", 10))
    for row in context.get("holdings_review", []) or []:
        add_symbol(symbols, row.get("ticker"))
    for row in (context.get("top_candidates", []) or [])[:top_candidate_limit]:
        if row.get("category") in {"A", "B", "D", "E"}:
            add_symbol(symbols, row.get("ticker"))
    for symbol in context.get("after_close_review", {}).get("news_watch_symbols", []) or []:
        add_symbol(symbols, symbol)
    for row in context.get("watchlist_review", {}).get("triggered", []) or []:
        add_symbol(symbols, row.get("ticker"))
    for row in context.get("watchlist_review", {}).get("missed", []) or []:
        add_symbol(symbols, row.get("ticker"))
    closed_loop = context.get("closed_loop", {}) if isinstance(context.get("closed_loop"), dict) else {}
    radar = closed_loop.get("radar", {}) if isinstance(closed_loop.get("radar"), dict) else {}
    for row in (radar.get("active_trend", []) or [])[:4]:
        add_symbol(symbols, row.get("ticker"))
    return symbols


def earnings_watch_symbols(context: dict[str, Any]) -> list[str]:
    symbols: list[str] = []
    for row in context.get("holdings_review", []) or []:
        add_symbol(symbols, row.get("ticker"))
    for row in context.get("top_candidates", []) or []:
        add_symbol(symbols, row.get("ticker"))
    for row in context.get("watchlist_review", {}).get("items", []) or []:
        add_symbol(symbols, row.get("ticker"))
    closed_loop = context.get("closed_loop", {}) if isinstance(context.get("closed_loop"), dict) else {}
    radar = closed_loop.get("radar", {}) if isinstance(closed_loop.get("radar"), dict) else {}
    for bucket in ["active_trend", "speculative_events", "sector_rotation"]:
        for row in radar.get(bucket, []) or []:
            add_symbol(symbols, row.get("ticker"))
    return symbols


def price_lookup(context: dict[str, Any]) -> dict[str, float]:
    lookup: dict[str, float] = {}
    for key, symbol_key in [("holdings_review", "ticker"), ("top_candidates", "ticker"), ("market_dashboard", "symbol")]:
        for row in context.get(key, []) or []:
            symbol = str(row.get(symbol_key, "")).strip().upper()
            if not symbol:
                continue
            try:
                lookup[symbol] = float(row.get("close"))
            except (TypeError, ValueError):
                continue
    return lookup


def add_symbol(symbols: list[str], value: Any) -> None:
    symbol = str(value or "").strip().upper()
    if not symbol or symbol in {"CASH", "MARGIN_BALANCE"}:
        return
    if symbol.startswith("^") or "=" in symbol:
        return
    if symbol not in symbols:
        symbols.append(symbol)


def scan_price_lookup(scan: ScanResult) -> dict[str, float]:
    lookup: dict[str, float] = {}
    for frame, symbol_key in [(scan.holdings_review, "ticker"), (scan.candidates, "ticker"), (scan.market_dashboard, "symbol")]:
        if frame is None or frame.empty or symbol_key not in frame or "close" not in frame:
            continue
        for row in frame[[symbol_key, "close"]].itertuples(index=False):
            symbol = str(getattr(row, symbol_key)).strip().upper()
            if not symbol:
                continue
            try:
                lookup[symbol] = float(getattr(row, "close"))
            except (TypeError, ValueError):
                continue
    return lookup

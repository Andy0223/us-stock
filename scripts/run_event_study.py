#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import sys
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from us_stock_signal.config import load_config, resolve_path
from us_stock_signal.event_study import run_event_study


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run US stock swing-rally event study.")
    parser.add_argument("--config", default="config/default_config.json", help="Path to JSON config file.")
    parser.add_argument("--as-of", default=date.today().isoformat(), help="Study date in YYYY-MM-DD format.")
    parser.add_argument("--lookback-days", type=int, default=None, help="Trading-day lookback window.")
    parser.add_argument("--horizon-days", type=int, default=None, help="Forward rally detection window.")
    parser.add_argument("--min-forward-return", type=float, default=None, help="Big-rally threshold, e.g. 0.30.")
    parser.add_argument("--min-forward-excess-return", type=float, default=None, help="Excess return threshold vs SPY.")
    parser.add_argument("--min-avg-dollar-volume", type=float, default=None, help="Minimum 20-day average dollar volume.")
    parser.add_argument("--sample-step-days", type=int, default=None, help="Observation sampling interval.")
    parser.add_argument("--max-symbols", type=int, default=None, help="Limit symbols after liquidity sorting.")
    parser.add_argument(
        "--selection-scores",
        default=None,
        help="Optional candidate_scores CSV used for current liquidity sorting before historical scan.",
    )
    parser.add_argument("--refresh-price-cache", action="store_true", help="Force price-cache refresh.")
    parser.add_argument("--output-dir", default=None, help="Output directory override.")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    config, config_dir = load_config(args.config)
    if args.output_dir:
        config["output_dir"] = args.output_dir
    as_of = date.fromisoformat(args.as_of)

    result = run_event_study(
        config,
        config_dir,
        as_of,
        lookback_days=args.lookback_days,
        horizon_days=args.horizon_days,
        min_forward_return=args.min_forward_return,
        min_forward_excess_return=args.min_forward_excess_return,
        min_avg_dollar_volume=args.min_avg_dollar_volume,
        sample_step_days=args.sample_step_days,
        max_symbols=args.max_symbols,
        selection_scores_path=args.selection_scores,
        force_refresh=args.refresh_price_cache,
    )

    output_dir = resolve_path(config.get("output_dir", "outputs"), config_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = as_of.strftime("%Y%m%d")
    report_path = output_dir / f"event_study_report_{suffix}.md"
    events_path = output_dir / f"event_study_events_{suffix}.csv"
    factor_path = output_dir / f"event_study_factor_lift_{suffix}.csv"
    sector_path = output_dir / f"event_study_sector_summary_{suffix}.csv"
    regime_path = output_dir / f"event_study_regime_summary_{suffix}.csv"
    watchlist_path = output_dir / f"event_study_current_factor_watchlist_{suffix}.csv"

    report_path.write_text(result.report_text, encoding="utf-8")
    result.events.to_csv(events_path, index=False)
    result.factor_lift.to_csv(factor_path, index=False)
    result.sector_summary.to_csv(sector_path, index=False)
    result.regime_summary.to_csv(regime_path, index=False)
    result.current_watchlist.to_csv(watchlist_path, index=False)

    print(result.report_text)
    print(f"Wrote report: {report_path}")
    print(f"Wrote events: {events_path}")
    print(f"Wrote factor lift: {factor_path}")
    print(f"Wrote sector summary: {sector_path}")
    print(f"Wrote regime summary: {regime_path}")
    print(f"Wrote current factor watchlist: {watchlist_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import logging
from datetime import date

from .config import load_config
from .env import load_env_file
from .pipeline import run_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="US stock swing trading strategy runner.")
    parser.add_argument("--config", default="config/default_config.json", help="Path to JSON config file.")
    parser.add_argument("--universe", help="Override universe CSV path.")
    parser.add_argument("--holdings", help="Override holdings CSV path.")
    parser.add_argument("--strategy-prompt", help="Override strategy prompt markdown path.")
    parser.add_argument("--output-dir", help="Override output directory.")
    parser.add_argument("--env-file", help="Optional env file to load.")
    parser.add_argument("--as-of", help="Run date in YYYY-MM-DD format.")
    parser.add_argument("--mode", choices=["premarket", "after_close"], help="Report mode.")
    parser.add_argument("--force", action="store_true", help="Run even on non-trading days.")
    parser.add_argument("--refresh-price-cache", action="store_true", help="Force refresh cached price data.")
    parser.add_argument("--skip-ai", action="store_true", help="Use local fallback report instead of OpenAI.")
    parser.add_argument("--send-telegram", action="store_true", help="Send report to Telegram.")
    parser.add_argument("--no-telegram", action="store_true", help="Disable Telegram send.")
    parser.add_argument("--dry-run", action="store_true", help="Write files but do not send Telegram.")
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
    if args.env_file:
        load_env_file(args.env_file)
    apply_overrides(config, args)
    as_of = date.fromisoformat(args.as_of) if args.as_of else None

    try:
        result = run_pipeline(
            config,
            config_dir,
            mode=args.mode,
            as_of=as_of,
            force=args.force,
            skip_ai=args.skip_ai,
            dry_run=args.dry_run,
        )
    except Exception:
        logging.getLogger(__name__).exception("US swing run failed")
        return 1

    print(result.report.text)
    print()
    if result.report_path:
        print(f"Wrote report: {result.report_path}")
    if result.context_path:
        print(f"Wrote context: {result.context_path}")
    if result.candidates_path:
        print(f"Wrote candidates: {result.candidates_path}")
    print(f"Provider: {result.report.provider_status}")
    print(f"Telegram sent: {result.telegram_sent}")
    if result.telegram_error:
        print(f"Telegram error: {result.telegram_error}")
    return 0


def apply_overrides(config: dict, args: argparse.Namespace) -> None:
    if args.universe:
        config["universe_path"] = args.universe
    if args.holdings:
        config["holdings_path"] = args.holdings
    if args.strategy_prompt:
        config["strategy_prompt_path"] = args.strategy_prompt
    if args.output_dir:
        config["output_dir"] = args.output_dir
    if args.mode:
        config["mode"] = args.mode
    if args.refresh_price_cache:
        config.setdefault("data_fetch", {}).setdefault("cache", {})["force_refresh"] = True
    if args.send_telegram:
        config.setdefault("telegram", {})["enabled"] = True
    if args.no_telegram or args.dry_run:
        config.setdefault("telegram", {})["enabled"] = False

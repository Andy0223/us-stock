from __future__ import annotations

import argparse
import logging
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from us_stock_signal.env import load_env_file
from us_stock_signal.firstrade_import import format_import_message, import_firstrade_positions
from us_stock_signal.notify import send_telegram_message


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Import a Firstrade positions CSV into data/holdings.csv.")
    parser.add_argument("--csv", dest="source_csv", help="Explicit Firstrade CSV path.")
    parser.add_argument("--inbox", default="data/inbox/firstrade", help="Folder to auto-detect the newest CSV.")
    parser.add_argument("--output", default="data/holdings.csv", help="Holdings CSV output path.")
    parser.add_argument("--archive-dir", default="data/raw/firstrade", help="Raw archive folder.")
    parser.add_argument("--cash", type=float, default=None, help="Override USD cash row.")
    parser.add_argument("--allow-missing", action="store_true", help="Exit successfully when no CSV is found.")
    parser.add_argument("--as-of", help="Import date in YYYY-MM-DD format.")
    parser.add_argument("--env-file", default=".env", help="Optional env file.")
    parser.add_argument("--send-telegram", action="store_true", help="Send import summary to Telegram.")
    parser.add_argument("--no-telegram", action="store_true", help="Disable Telegram send.")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser


def main() -> int:
    args = build_parser().parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    env_path = Path(args.env_file).expanduser()
    if args.env_file and env_path.exists():
        load_env_file(env_path)

    result = import_firstrade_positions(
        source=args.source_csv,
        inbox_dir=args.inbox,
        output_path=args.output,
        archive_dir=args.archive_dir,
        cash=args.cash,
        allow_missing=args.allow_missing,
        as_of=date.fromisoformat(args.as_of) if args.as_of else None,
    )
    message = format_import_message(result)
    print(message)
    if args.send_telegram and not args.no_telegram:
        send_telegram_message(
            {
                "bot_token_env": "US_STOCK_TELEGRAM_BOT_TOKEN",
                "chat_id_env": "US_STOCK_TELEGRAM_CHAT_ID",
            },
            message,
        )
        print("Telegram notification sent.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

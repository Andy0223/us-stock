from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from us_stock_signal.universe import update_us_universe


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build the US tradable universe from Nasdaq Trader symbol directories.")
    parser.add_argument("--output", default="data/universe_us_all.csv", help="Universe CSV output path.")
    parser.add_argument("--raw-dir", default="data/raw/universe", help="Raw downloaded symbol directory archive.")
    parser.add_argument("--include-etfs", action="store_true", help="Include ETFs in the generated universe.")
    parser.add_argument("--min-symbols", type=int, default=1000, help="Refuse to overwrite output below this row count.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    result = update_us_universe(
        output_path=args.output,
        raw_dir=args.raw_dir,
        include_etfs=args.include_etfs,
        min_symbols=args.min_symbols,
    )
    print(f"Wrote universe: {result.output_path}")
    print(f"Rows: {result.row_count}")
    print(f"Skipped: {result.skipped_count}")
    print(f"Raw archive: {result.raw_dir}")
    for exchange, count in sorted(result.source_counts.items(), key=lambda item: (-item[1], item[0])):
        print(f"- {exchange}: {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

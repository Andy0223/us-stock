#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "$0")" && pwd -P)/cron_bootstrap.sh"

ARGS=(
  -B
  scripts/import_firstrade_positions.py
  --inbox "${FIRSTRADE_INBOX_DIR:-data/inbox/firstrade}"
  --output "${FIRSTRADE_HOLDINGS_OUTPUT:-data/holdings.csv}"
  --archive-dir "${FIRSTRADE_ARCHIVE_DIR:-data/raw/firstrade}"
  --allow-missing
)

if [[ -f ".env" ]]; then
  ARGS+=(--env-file .env)
fi

if [[ -n "${FIRSTRADE_CSV:-}" ]]; then
  ARGS+=(--csv "$FIRSTRADE_CSV")
fi

if [[ -n "${FIRSTRADE_CASH_OVERRIDE:-}" ]]; then
  ARGS+=(--cash "$FIRSTRADE_CASH_OVERRIDE")
fi

if [[ -n "${US_SWING_AS_OF:-}" ]]; then
  ARGS+=(--as-of "$US_SWING_AS_OF")
fi

if [[ "${FIRSTRADE_SYNC_SEND_TELEGRAM:-0}" == "1" ]]; then
  ARGS+=(--send-telegram)
else
  ARGS+=(--no-telegram)
fi

run_python "${ARGS[@]}" "$@"

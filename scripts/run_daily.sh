#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "$0")" && pwd -P)/cron_bootstrap.sh"

CONFIG_PATH="${CONFIG_PATH:-config/default_config.json}"
UNIVERSE_PATH="${UNIVERSE_PATH:-data/universe_us_all.csv}"
MODE="${US_SWING_MODE:-premarket}"

UPDATE_UNIVERSE="${US_SWING_UPDATE_UNIVERSE:-auto}"
if [[ "$UPDATE_UNIVERSE" == "auto" ]]; then
  if [[ "$UNIVERSE_PATH" == "data/universe_us_all.csv" || "$UNIVERSE_PATH" == */data/universe_us_all.csv ]]; then
    UPDATE_UNIVERSE="1"
  else
    UPDATE_UNIVERSE="0"
  fi
fi
if [[ "$UPDATE_UNIVERSE" == "1" ]]; then
  UNIVERSE_UPDATE_ARGS=(--output "$UNIVERSE_PATH")
  if [[ "${US_SWING_INCLUDE_ETFS:-0}" == "1" ]]; then
    UNIVERSE_UPDATE_ARGS+=(--include-etfs)
  fi
  if ! run_python -B scripts/update_us_universe.py "${UNIVERSE_UPDATE_ARGS[@]}"; then
    if [[ -f "$UNIVERSE_PATH" ]]; then
      echo "US universe update failed; continuing with existing $UNIVERSE_PATH." >&2
    else
      echo "US universe update failed and $UNIVERSE_PATH does not exist." >&2
      exit 1
    fi
  fi
fi

ENV_ARGS=()
if [[ -f ".env" ]]; then
  ENV_ARGS=(--env-file .env)
fi

ARGS=(
  -B
  -m us_stock_signal
  --config "$CONFIG_PATH"
  --universe "$UNIVERSE_PATH"
  --mode "$MODE"
  "${ENV_ARGS[@]}"
)

if [[ "${US_SWING_FORCE:-0}" == "1" ]]; then
  ARGS+=(--force)
fi

if [[ "${US_SWING_FORCE_REFRESH:-0}" == "1" ]]; then
  ARGS+=(--refresh-price-cache)
fi

USE_OPENAI_API="${US_SWING_USE_OPENAI_API:-0}"
if [[ "${US_SWING_SKIP_AI:-0}" == "1" || "$USE_OPENAI_API" != "1" ]]; then
  ARGS+=(--skip-ai)
fi

if [[ -n "${US_SWING_AS_OF:-}" ]]; then
  ARGS+=(--as-of "$US_SWING_AS_OF")
fi

if [[ "${US_SWING_DRY_RUN:-0}" == "1" ]]; then
  ARGS+=(--dry-run)
fi

SEND_TELEGRAM="${US_SWING_SEND_TELEGRAM:-auto}"
if [[ "$SEND_TELEGRAM" == "auto" ]]; then
  if [[ -n "${US_STOCK_TELEGRAM_BOT_TOKEN:-}" && -n "${US_STOCK_TELEGRAM_CHAT_ID:-}" ]]; then
    SEND_TELEGRAM="1"
  elif [[ -f ".env" ]] && grep -Eq '^US_STOCK_TELEGRAM_BOT_TOKEN=.+' .env && grep -Eq '^US_STOCK_TELEGRAM_CHAT_ID=.+' .env; then
    SEND_TELEGRAM="1"
  else
    SEND_TELEGRAM="0"
  fi
fi

if [[ "$SEND_TELEGRAM" == "1" ]]; then
  ARGS+=(--send-telegram)
else
  ARGS+=(--no-telegram)
fi

run_python "${ARGS[@]}" "$@"

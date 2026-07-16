from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any


DEFAULT_CONFIG: dict[str, Any] = {
    "universe_path": "data/universe_us_all.csv",
    "holdings_path": "data/holdings.csv",
    "strategy_prompt_path": "prompts/us_swing_strategy.md",
    "output_dir": "outputs",
    "mode": "premarket",
    "market": {
        "timezone": "America/New_York",
        "skip_non_trading_days": True,
        "benchmark": "SPY",
    },
    "lookback_days": 320,
    "data_fetch": {
        "batch_size": 40,
        "pause_seconds": 1.0,
        "max_retries": 3,
        "retry_pause_seconds": 10.0,
        "timeout_seconds": 45.0,
        "threads": True,
        "cache": {
            "enabled": True,
            "dir": "data/price_cache",
            "force_refresh": False,
            "max_stale_calendar_days": 5,
            "min_coverage_ratio": 0.70,
            "use_stale_on_failure": True,
        },
    },
    "market_dashboard": {
        "SPY": "S&P 500",
        "QQQ": "Nasdaq 100",
        "SMH": "Semiconductors",
        "IWM": "Russell 2000",
        "^VIX": "VIX",
        "TLT": "20Y Treasury",
        "XLE": "Energy",
        "XLU": "Utilities",
        "DX-Y.NYB": "US Dollar Index",
        "^TNX": "10Y Treasury Yield",
        "^IRX": "13W Treasury Yield",
        "CL=F": "WTI Crude",
        "BZ=F": "Brent Crude",
        "GC=F": "Gold",
        "HG=F": "Copper",
        "NG=F": "Natural Gas",
        "BTC-USD": "Bitcoin",
        "TSM": "TSMC ADR",
        "ASML": "ASML ADR",
    },
    "structural_theme_scores": {
        "US common stocks": 0.55,
        "ETF": 0.45,
        "Semiconductors and hardware": 0.85,
        "Software and technology": 0.78,
        "Healthcare and biotech": 0.58,
        "Financials": 0.55,
        "Energy and utilities": 0.58,
        "Industrials": 0.62,
        "Consumer": 0.50,
        "Real estate": 0.48,
        "Materials": 0.56,
        "ADR and overseas listings": 0.58,
        "AI semiconductors": 1.00,
        "AI networking": 0.95,
        "AI data center infrastructure": 0.95,
        "Power grid and energy equipment": 0.92,
        "Data center REIT and digital infrastructure": 0.82,
        "Industrial infrastructure and materials": 0.82,
        "Healthcare and life science tools": 0.72,
        "Defense aerospace and space": 0.80,
        "Financial infrastructure": 0.70,
        "Consumer brand repair": 0.62,
        "Software cybersecurity and AI software": 0.78,
        "Overseas supply chain ADR": 0.80,
    },
    "high_risk_sectors": [
        "Defense aerospace and space",
        "Software cybersecurity and AI software",
        "Healthcare and biotech",
        "Software and technology",
    ],
    "risk": {
        "red_vix": 25.0,
        "yellow_vix": 20.0,
        "max_single_stock_weight": 0.12,
        "max_sector_weight": 0.30,
        "high_beta_first_tranche_pct": 0.10,
        "normal_first_tranche_pct": 0.25,
        "margin": {
            "enabled": True,
            "allowed_risk_lights": ["green"],
            "allowed_categories": ["A"],
            "max_negative_cash_weight": 0.20,
            "max_single_margin_trade_pct": 0.05,
            "max_new_margin_exposure_pct": 0.08,
        },
    },
    "scanner": {
        "top_candidates": 40,
        "min_avg_dollar_volume": 25_000_000,
        "overextended_return_1d": 0.10,
        "overextended_return_20d": 0.35,
        "overextended_pct_above_ma20": 0.15,
    },
    "watchlist": {
        "enabled": True,
        "path": "data/watchlist_next_day.csv",
    },
    "options_review": {
        "enabled": True,
        "max_symbols": 14,
        "top_candidate_symbols": 10,
        "max_expirations": 3,
        "min_days_to_expiry": 3,
        "max_days_to_expiry": 45,
        "near_atm_pct": 0.05,
        "min_total_contract_volume": 500,
        "min_near_atm_volume": 100,
        "bullish_call_put_ratio": 1.50,
        "bearish_put_call_ratio": 1.30,
        "high_iv_threshold": 0.65,
        "max_alerts": 8,
        "pause_seconds": 0.05,
    },
    "ai_research": {
        "enabled": True,
        "provider": "openai",
        "api_key_env": "OPENAI_API_KEY",
        "base_url_env": "OPENAI_BASE_URL",
        "model_env": "OPENAI_MODEL",
        "reasoning_effort_env": "OPENAI_REASONING_EFFORT",
        "default_model": "gpt-5.6",
        "default_reasoning_effort": "xhigh",
        "timeout_seconds": 60,
        "temperature": 0.2,
        "max_strategy_chars": 70000,
    },
    "telegram": {
        "enabled": False,
        "bot_token_env": "US_STOCK_TELEGRAM_BOT_TOKEN",
        "chat_id_env": "US_STOCK_TELEGRAM_CHAT_ID",
    },
}


def deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path: str | Path | None = None) -> tuple[dict[str, Any], Path]:
    if path is None:
        return copy.deepcopy(DEFAULT_CONFIG), Path.cwd()

    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as fh:
        user_config = json.load(fh)
    return deep_merge(DEFAULT_CONFIG, user_config), config_path.parent


def resolve_path(path_value: str | Path, config_dir: Path) -> Path:
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path

    candidates = [
        Path.cwd() / path,
        config_dir / path,
        config_dir.parent / path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    if config_dir.name == "config":
        return (config_dir.parent / path).resolve()
    return (config_dir / path).resolve()

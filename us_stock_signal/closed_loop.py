from __future__ import annotations

import math
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

from .config import resolve_path


DEFAULT_FACTOR_LABELS = {
    "high_atr": "高波動",
    "deep_reversal_risk": "深跌反轉/軋空",
    "sector_rotation_plus_stock_rs": "產業輪動 + 個股相對強",
    "breakout_with_volume": "放量突破",
    "relative_strength_60d": "60日相對SPY強",
    "relative_strength_20d": "20日相對SPY強",
    "sector_top_quartile_20d": "產業20日強度前25%",
    "sector_top_quartile_60d": "產業60日強度前25%",
    "volume_dryup": "量縮整理",
    "volume_expansion": "當日放量",
}


def build_closed_loop_context(
    config: dict[str, Any],
    config_dir: Path,
    base_context: dict[str, Any],
    as_of: date,
    mode: str,
    *,
    candidate_frame: pd.DataFrame | None = None,
    sector_score_frame: pd.DataFrame | None = None,
    market_dashboard_frame: pd.DataFrame | None = None,
) -> dict[str, Any]:
    cfg = dict(config.get("closed_loop", {}))
    if not bool(cfg.get("enabled", True)):
        return {"enabled": False, "note": "closed loop disabled"}

    output_dir = resolve_path(config.get("output_dir", "outputs"), config_dir)
    factor_lift, factor_source = load_factor_lift(cfg, config_dir, output_dir)
    candidates = candidate_frame.copy() if candidate_frame is not None else pd.DataFrame(base_context.get("top_candidates", []) or [])
    all_candidates_path = output_dir / f"candidate_scores_{as_of.strftime('%Y%m%d')}.csv"
    if candidate_frame is None and all_candidates_path.exists():
        try:
            candidates = pd.read_csv(all_candidates_path)
        except Exception:
            pass
    if candidates.empty:
        candidates = pd.DataFrame(base_context.get("top_candidates", []) or [])

    radar = build_factor_radar(
        candidates,
        sector_score_frame.copy()
        if sector_score_frame is not None
        else pd.DataFrame(base_context.get("sector_scores", []) or []),
        market_dashboard_frame.copy()
        if market_dashboard_frame is not None
        else pd.DataFrame(base_context.get("market_dashboard", []) or []),
        factor_lift,
        cfg,
    )
    feedback = build_feedback_summary(base_context, mode)
    learning = build_learning_summary(factor_lift)

    return {
        "enabled": True,
        "as_of": as_of.isoformat(),
        "mode": mode,
        "stage": "premarket_plan" if mode == "premarket" else "after_close_review",
        "factor_source": str(factor_source) if factor_source else "",
        "learning": learning,
        "feedback": feedback,
        "radar": radar,
        "loop_note": "歷史因子 -> 每日雷達 -> 盤後檢討 -> 隔日追蹤清單；AI 只做摘要與排序，不直接覆寫風控規則。",
    }


def load_factor_lift(
    cfg: dict[str, Any],
    config_dir: Path,
    output_dir: Path,
) -> tuple[pd.DataFrame, Path | None]:
    configured = str(cfg.get("factor_lift_path", "") or "").strip()
    if configured:
        path = resolve_path(configured, config_dir)
        if path.exists():
            return pd.read_csv(path), path

    pattern = str(cfg.get("factor_lift_glob", "event_study_factor_lift_*.csv"))
    matches = sorted(output_dir.glob(pattern))
    if not matches:
        return pd.DataFrame(), None
    path = matches[-1]
    try:
        return pd.read_csv(path), path
    except Exception:
        return pd.DataFrame(), path


def build_factor_radar(
    candidates: pd.DataFrame,
    sector_scores: pd.DataFrame,
    market_dashboard: pd.DataFrame,
    factor_lift: pd.DataFrame,
    cfg: dict[str, Any],
) -> dict[str, Any]:
    if candidates.empty or factor_lift.empty:
        return {
            "active_trend": [],
            "speculative_events": [],
            "sector_rotation": [],
            "all": [],
            "source_status": {
                "candidate_rows": len(candidates),
                "factor_rows": len(factor_lift),
                "note": "missing candidate or factor data",
            },
        }

    useful = useful_factors(
        factor_lift,
        min_lift=float(cfg.get("min_factor_lift", 1.20)),
        min_events=int(cfg.get("min_factor_events", 100)),
    )
    if useful.empty:
        return {
            "active_trend": [],
            "speculative_events": [],
            "sector_rotation": [],
            "all": [],
            "source_status": {
                "candidate_rows": len(candidates),
                "factor_rows": len(factor_lift),
                "note": "no useful factor survived thresholds",
            },
        }

    market = market_lookup(market_dashboard)
    sector_ranks = sector_rank_lookup(sector_scores)
    rows: list[dict[str, Any]] = []
    for row in candidates.to_dict("records"):
        factors = active_factors_for_row(row, sector_ranks, market)
        matched = [factor for factor in factors if factor in set(useful["factor"].astype(str))]
        if not matched:
            continue
        factor_score = 0.0
        factor_lift_map = useful.set_index("factor")["lift"].to_dict()
        factor_event_map = useful.set_index("factor")["events"].to_dict()
        for factor in matched:
            factor_score += max(0.0, math.log(float(factor_lift_map.get(factor, 1.0)))) * 100.0
        risk_tier = risk_tier_for(row, matched)
        actionability_score = factor_score * actionability_multiplier(row, matched)
        setup_type = setup_type_for(matched)
        rows.append(
            {
                "ticker": str(row.get("ticker", "")).upper(),
                "name": row.get("name", ""),
                "sector": row.get("sector", ""),
                "category": row.get("category", ""),
                "action_bias": row.get("action_bias", ""),
                "setup_type": setup_type,
                "risk_tier": risk_tier,
                "factor_score": round(factor_score, 2),
                "actionability_score": round(actionability_score, 2),
                "active_factors": ", ".join(factor_label(factor) for factor in matched),
                "active_factor_ids": ", ".join(matched),
                "factor_events": int(sum(float(factor_event_map.get(factor, 0)) for factor in matched)),
                "close": safe_float(row.get("close")),
                "return_1d": safe_float(row.get("return_1d")),
                "return_5d": safe_float(row.get("return_5d")),
                "return_20d": safe_float(row.get("return_20d")),
                "return_60d": safe_float(row.get("return_60d")),
                "relative_return_20d": safe_float(row.get("relative_return_20d")),
                "volume_ratio_20d": safe_float(row.get("volume_ratio_20d")),
                "rsi14": safe_float(row.get("rsi14")),
                "atr_pct": safe_float(row.get("atr_pct")),
                "entry_low": safe_float(row.get("entry_low")),
                "entry_high": safe_float(row.get("entry_high")),
                "stop_loss": safe_float(row.get("stop_loss")),
                "no_chase_above": safe_float(row.get("no_chase_above")),
                "reason": radar_reason(row, matched),
            }
        )

    all_rows = sorted(rows, key=lambda item: (item["actionability_score"], item["factor_score"]), reverse=True)
    min_actionability = float(cfg.get("min_actionability_score", 45.0))
    filtered = [
        row
        for row in all_rows
        if row["actionability_score"] >= min_actionability and passes_quality_filter(row, cfg)
    ]
    active = [row for row in filtered if row["risk_tier"] != "speculative"]
    speculative = [row for row in filtered if row["risk_tier"] == "speculative"]
    sector_rotation = [row for row in filtered if "sector_rotation_plus_stock_rs" in row["active_factor_ids"]]
    max_trend = int(cfg.get("max_trend_items", 8))
    max_spec = int(cfg.get("max_speculative_items", 6))
    max_sector = int(cfg.get("max_sector_items", 6))
    return {
        "active_trend": active[:max_trend],
        "speculative_events": speculative[:max_spec],
        "sector_rotation": sector_rotation[:max_sector],
        "all": filtered[: int(cfg.get("max_all_items", 30))],
        "source_status": {
            "candidate_rows": len(candidates),
            "factor_rows": len(factor_lift),
            "useful_factor_rows": len(useful),
            "radar_rows": len(filtered),
            "min_actionability_score": min_actionability,
        },
    }


def passes_quality_filter(row: dict[str, Any], cfg: dict[str, Any]) -> bool:
    close = safe_float(row.get("close"))
    if close < float(cfg.get("min_close", 5.0)):
        return False

    risk_tier = str(row.get("risk_tier", ""))
    volume_ratio = safe_float(row.get("volume_ratio_20d"), 0.0)
    category = str(row.get("category", ""))
    if risk_tier == "speculative":
        if not bool(cfg.get("include_speculative", True)):
            return False
        return volume_ratio >= float(cfg.get("speculative_min_volume_ratio_20d", 0.50))

    allowed_categories = set(cfg.get("active_allowed_categories", ["A", "B", "D", "E"]))
    if category not in allowed_categories:
        return False
    return volume_ratio >= float(cfg.get("active_min_volume_ratio_20d", 0.35))


def useful_factors(factor_lift: pd.DataFrame, min_lift: float, min_events: int) -> pd.DataFrame:
    frame = factor_lift.copy()
    if frame.empty or "factor" not in frame:
        return pd.DataFrame()
    frame["lift"] = pd.to_numeric(frame.get("lift"), errors="coerce")
    frame["events"] = pd.to_numeric(frame.get("events"), errors="coerce")
    frame = frame[(frame["lift"] >= min_lift) & (frame["events"] >= min_events)]
    return frame.sort_values(["lift", "events"], ascending=False).reset_index(drop=True)


def active_factors_for_row(row: dict[str, Any], sector_ranks: dict[str, dict[str, float]], market: dict[str, float]) -> list[str]:
    factors: list[str] = []
    return_20d = safe_float(row.get("return_20d"))
    return_60d = safe_float(row.get("return_60d"))
    relative_20d = safe_float(row.get("relative_return_20d"))
    spy_60d = market.get("SPY_return_60d", 0.0)
    relative_60d = return_60d - spy_60d if is_finite(return_60d) and is_finite(spy_60d) else math.nan
    volume_ratio = safe_float(row.get("volume_ratio_20d"))
    atr_pct = safe_float(row.get("atr_pct"))
    drawdown_20 = safe_float(row.get("drawdown_from_20d_high"))
    pct_ma200 = safe_float(row.get("pct_from_ma200"))
    sector = str(row.get("sector", ""))
    sector_rank = sector_ranks.get(sector, {})
    sector_top_20 = sector_rank.get("rank_20d", 1.0) <= 0.25
    sector_top_60 = sector_rank.get("rank_60d", 1.0) <= 0.25

    if atr_pct >= 0.06:
        factors.append("high_atr")
    if pct_ma200 <= -0.30 and volume_ratio >= 1.5:
        factors.append("deep_reversal_risk")
    if relative_20d >= 0.05:
        factors.append("relative_strength_20d")
    if relative_60d >= 0.10:
        factors.append("relative_strength_60d")
    if sector_top_20:
        factors.append("sector_top_quartile_20d")
    if sector_top_60:
        factors.append("sector_top_quartile_60d")
    if sector_top_20 and relative_20d >= 0.05:
        factors.append("sector_rotation_plus_stock_rs")
    if drawdown_20 >= -0.005 and volume_ratio >= 1.5 and relative_20d >= 0.05:
        factors.append("breakout_with_volume")
    if volume_ratio <= 0.75 and safe_float(row.get("pct_from_ma50")) > 0:
        factors.append("volume_dryup")
    if volume_ratio >= 1.5:
        factors.append("volume_expansion")
    return factors


def market_lookup(market_dashboard: pd.DataFrame) -> dict[str, float]:
    result: dict[str, float] = {}
    if market_dashboard.empty or "symbol" not in market_dashboard:
        return result
    for row in market_dashboard.to_dict("records"):
        symbol = str(row.get("symbol", "")).upper()
        if symbol == "SPY":
            result["SPY_return_20d"] = safe_float(row.get("return_20d"), 0.0)
            result["SPY_return_60d"] = safe_float(row.get("return_60d"), 0.0)
    return result


def sector_rank_lookup(sector_scores: pd.DataFrame) -> dict[str, dict[str, float]]:
    if sector_scores.empty or "sector" not in sector_scores:
        return {}
    frame = sector_scores.copy()
    frame["avg_return_20d"] = pd.to_numeric(frame.get("avg_return_20d"), errors="coerce")
    frame["avg_return_60d"] = pd.to_numeric(frame.get("avg_return_60d", frame.get("avg_return_20d")), errors="coerce")
    frame["rank_20d"] = frame["avg_return_20d"].rank(pct=True, ascending=False)
    frame["rank_60d"] = frame["avg_return_60d"].rank(pct=True, ascending=False)
    return {
        str(row["sector"]): {
            "rank_20d": safe_float(row.get("rank_20d"), 1.0),
            "rank_60d": safe_float(row.get("rank_60d"), 1.0),
            "avg_return_20d": safe_float(row.get("avg_return_20d")),
            "avg_return_60d": safe_float(row.get("avg_return_60d")),
        }
        for row in frame.to_dict("records")
    }


def build_feedback_summary(context: dict[str, Any], mode: str) -> dict[str, Any]:
    watch = context.get("watchlist_review", {}) if isinstance(context.get("watchlist_review"), dict) else {}
    watch_summary = watch.get("summary", {}) if isinstance(watch.get("summary"), dict) else {}
    review = context.get("after_close_review", {}) if isinstance(context.get("after_close_review"), dict) else {}
    missed_buy = review.get("missed_buy_candidates", []) or []
    missed_sell = review.get("missed_sell_candidates", []) or []
    missed_add = review.get("missed_add_candidates", []) or []
    return {
        "mode": mode,
        "watchlist_active": int_or_zero(watch_summary.get("active_count")),
        "watchlist_triggered": int_or_zero(watch_summary.get("triggered_count")),
        "watchlist_missed": int_or_zero(watch_summary.get("missed_count")),
        "missed_buy_count": len(missed_buy),
        "missed_sell_count": len(missed_sell),
        "missed_add_count": len(missed_add),
        "next_action": feedback_next_action(watch_summary, missed_buy, missed_sell, missed_add),
    }


def build_learning_summary(factor_lift: pd.DataFrame) -> dict[str, Any]:
    if factor_lift.empty:
        return {"top_factors": [], "base_event_rate": math.nan}
    rows = []
    for row in factor_lift.head(8).to_dict("records"):
        rows.append(
            {
                "factor": str(row.get("factor", "")),
                "label": factor_label(row.get("factor")),
                "lift": safe_float(row.get("lift")),
                "event_rate": safe_float(row.get("event_rate")),
                "events": int_or_zero(row.get("events")),
            }
        )
    base = safe_float(factor_lift.iloc[0].get("base_event_rate")) if len(factor_lift) else math.nan
    return {"base_event_rate": base, "top_factors": rows}


def radar_reason(row: dict[str, Any], factors: list[str]) -> str:
    labels = [factor_label(factor) for factor in factors[:4]]
    category = str(row.get("category", ""))
    action = str(row.get("action_bias", ""))
    return f"{'、'.join(labels)}；模型分類 {category}/{action}。"


def setup_type_for(factors: list[str]) -> str:
    if "deep_reversal_risk" in factors:
        return "深跌反轉/軋空"
    if "breakout_with_volume" in factors:
        return "放量突破"
    if "sector_rotation_plus_stock_rs" in factors:
        return "產業輪動 + 個股相對強"
    if "volume_dryup" in factors and "relative_strength_20d" in factors:
        return "強勢量縮整理"
    if "relative_strength_60d" in factors:
        return "中期相對強勢"
    return "早期因子累積"


def risk_tier_for(row: dict[str, Any], factors: list[str]) -> str:
    if "deep_reversal_risk" in factors or "high_atr" in factors or bool(row.get("high_risk", False)):
        return "speculative"
    if "breakout_with_volume" in factors or "sector_rotation_plus_stock_rs" in factors:
        return "active"
    return "watch"


def actionability_multiplier(row: dict[str, Any], factors: list[str]) -> float:
    multiplier = 1.0
    if "deep_reversal_risk" in factors:
        multiplier *= 0.45
    if "high_atr" in factors:
        multiplier *= 0.65
    if bool(row.get("overextended", False)):
        multiplier *= 0.70
    if safe_float(row.get("rsi14")) > 80:
        multiplier *= 0.75
    if safe_float(row.get("volume_ratio_20d")) < 0.25:
        multiplier *= 0.75
    return multiplier


def feedback_next_action(
    watch_summary: dict[str, Any],
    missed_buy: list[dict[str, Any]],
    missed_sell: list[dict[str, Any]],
    missed_add: list[dict[str, Any]],
) -> str:
    if missed_sell:
        return "先處理該賣/該減碼，避免閉環只往買方偏。"
    if int_or_zero(watch_summary.get("missed_count")) > int_or_zero(watch_summary.get("triggered_count")):
        return "昨天追蹤清單有較多錯過項，隔日降低追價意願。"
    if missed_buy or missed_add:
        return "把錯過買點/加碼標的寫入隔日觀察，要求回測或突破確認。"
    return "沒有明確回饋錯誤，維持原風控。"


def closed_loop_watchlist_rows(context: dict[str, Any], valid_for: date, max_rows: int = 8) -> list[dict[str, Any]]:
    loop = context.get("closed_loop", {}) if isinstance(context.get("closed_loop"), dict) else {}
    radar = loop.get("radar", {}) if isinstance(loop.get("radar"), dict) else {}
    rows = []
    for priority, row in enumerate((radar.get("active_trend", []) or [])[:max_rows], start=6):
        rows.append(
            {
                "created_at": str(context.get("as_of", "")),
                "valid_for": valid_for.isoformat(),
                "ticker": str(row.get("ticker", "")).upper(),
                "name": row.get("name", ""),
                "action": "factor_watch",
                "priority": priority,
                "reason": row.get("reason", "閉環早期因子雷達。"),
                "trigger_low": row.get("entry_low", ""),
                "trigger_high": row.get("entry_high", ""),
                "stop_loss": row.get("stop_loss", ""),
                "no_chase_above": row.get("no_chase_above", ""),
                "source_section": "closed_loop_factor_radar",
                "status": "active",
            }
        )
    return rows


def radar_to_frame(loop_context: dict[str, Any]) -> pd.DataFrame:
    radar = loop_context.get("radar", {}) if isinstance(loop_context.get("radar"), dict) else {}
    rows = []
    for bucket in ["active_trend", "speculative_events", "sector_rotation", "all"]:
        for row in radar.get(bucket, []) or []:
            item = dict(row)
            item["bucket"] = bucket
            rows.append(item)
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows)
    return frame.drop_duplicates(subset=["bucket", "ticker"]).reset_index(drop=True)


def factor_label(value: Any) -> str:
    return DEFAULT_FACTOR_LABELS.get(str(value), str(value or ""))


def safe_float(value: Any, default: float = math.nan) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if math.isfinite(parsed) else default


def is_finite(value: Any) -> bool:
    return math.isfinite(safe_float(value))


def int_or_zero(value: Any) -> int:
    try:
        parsed = int(float(value))
    except (TypeError, ValueError):
        return 0
    return parsed

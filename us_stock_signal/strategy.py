from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import requests

from .config import resolve_path


@dataclass
class StrategyReport:
    as_of: date
    mode: str
    text: str
    provider_status: str


def generate_strategy_report(
    config: dict[str, Any],
    config_dir: Path,
    context: dict[str, Any],
    mode: str,
    as_of: date,
    skip_ai: bool = False,
) -> StrategyReport:
    ai_cfg = config.get("ai_research", {})
    if skip_ai or not bool(ai_cfg.get("enabled", True)):
        text = fallback_report(context, mode, "ai_disabled")
        return StrategyReport(as_of=as_of, mode=mode, text=text, provider_status="ai_disabled")

    prompt_path = resolve_path(config.get("strategy_prompt_path", ""), config_dir)
    strategy_text, prompt_status = load_strategy_prompt(prompt_path, int(ai_cfg.get("max_strategy_chars", 70000)))
    if not strategy_text:
        text = fallback_report(context, mode, prompt_status)
        return StrategyReport(as_of=as_of, mode=mode, text=text, provider_status=prompt_status)

    prompt = build_prompt(strategy_text, context, mode)
    text, status = call_openai(ai_cfg, prompt)
    if not text:
        text = fallback_report(context, mode, status)
    return StrategyReport(as_of=as_of, mode=mode, text=text, provider_status=status)


def load_strategy_prompt(path: Path, max_chars: int) -> tuple[str, str]:
    if not str(path):
        return "", "missing_strategy_prompt_path"
    if not path.exists():
        return "", f"strategy_prompt_not_found:{path}"
    text = path.read_text(encoding="utf-8").strip()
    if max_chars > 0 and len(text) > max_chars:
        text = text[:max_chars] + "\n\n[策略 prompt 已依設定截斷；請提高 max_strategy_chars 以包含全文。]"
    return text, "ok"


def build_prompt(strategy_text: str, context: dict[str, Any], mode: str) -> str:
    context_json = json.dumps(sanitize_for_json(context), ensure_ascii=False, indent=2)
    mode_name = "盤前策略" if mode == "premarket" else "盤後覆盤"
    return (
        f"以下是完整策略規格，請嚴格遵守。這次任務是產生「{mode_name}」。\n\n"
        f"{strategy_text}\n\n"
        "重要限制：\n"
        "1. 只能根據下方 JSON context 推論，不得假造財報、新聞、法人、估值或即時盤前資料。\n"
        "2. context 來自本專案設定的美股 universe，不是所有美股逐檔全市場掃描；若覆蓋不足，必須明說。\n"
        "3. fundamental_score_20 若只有 8 分，代表基本面尚未由資料驗證，不能把它說成已驗證。\n"
        "4. 請先處理持股與現金，再處理新候選。\n"
        "5. 輸出使用繁體中文，給出可執行價位、比例、失效條件與隔日雙向劇本。\n"
        "6. 如果任務是盤後覆盤，必須明確檢討該買未買、該賣未賣、該加碼未加、異常漲跌、產業輪動與 news_review 中的消息提醒。\n"
        "7. 融資不是絕對禁止；必須依 context.market_state.margin_policy 判斷。只有 green 風險燈號、A 類標的、未超過負現金上限、且非追高時，才可討論小額受控融資。\n"
        "8. options_review 是免費 yfinance 期權鏈摘要，只能當確認/警訊，不能當成完整 options flow 或內線訊號。\n"
        "9. 不得使用保證獲利、必漲、一定會漲、無風險等字眼。\n\n"
        f"JSON context:\n{context_json}"
    )


def call_openai(ai_cfg: dict[str, Any], prompt: str) -> tuple[str | None, str]:
    if str(ai_cfg.get("provider", "openai")).lower() != "openai":
        return None, "provider_disabled"
    api_key = os.getenv(str(ai_cfg.get("api_key_env", "OPENAI_API_KEY")), "")
    if not api_key:
        return None, "missing_openai_api_key"

    base_url = os.getenv(str(ai_cfg.get("base_url_env", "OPENAI_BASE_URL")), "").strip()
    base_url = (base_url or "https://api.openai.com/v1").rstrip("/")
    model = os.getenv(str(ai_cfg.get("model_env", "OPENAI_MODEL")), str(ai_cfg.get("default_model", "gpt-4o-mini")))
    reasoning_effort = os.getenv(
        str(ai_cfg.get("reasoning_effort_env", "OPENAI_REASONING_EFFORT")),
        str(ai_cfg.get("default_reasoning_effort", "")),
    ).strip()
    if model.startswith("gpt-5") or reasoning_effort:
        return call_openai_responses(ai_cfg, base_url, api_key, model, reasoning_effort, prompt)

    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "你是嚴謹、風控優先的美股波段交易研究員，只能根據使用者提供的資料推論。",
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": float(ai_cfg.get("temperature", 0.2)),
    }
    try:
        response = requests.post(
            f"{base_url}/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json=payload,
            timeout=float(ai_cfg.get("timeout_seconds", 60)),
        )
        response.raise_for_status()
        data = response.json()
        content = data["choices"][0]["message"]["content"]
        return str(content).strip(), f"openai:{model}"
    except Exception as exc:  # pragma: no cover - provider/network dependent
        return None, f"openai_error:{provider_error_summary(exc)}"


def call_openai_responses(
    ai_cfg: dict[str, Any],
    base_url: str,
    api_key: str,
    model: str,
    reasoning_effort: str,
    prompt: str,
) -> tuple[str | None, str]:
    payload: dict[str, Any] = {
        "model": model,
        "instructions": "你是嚴謹、風控優先的美股波段交易研究員，只能根據使用者提供的資料推論。",
        "input": prompt,
    }
    if reasoning_effort:
        payload["reasoning"] = {"effort": reasoning_effort}
    try:
        response = requests.post(
            f"{base_url}/responses",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json=payload,
            timeout=float(ai_cfg.get("timeout_seconds", 60)),
        )
        response.raise_for_status()
        data = response.json()
        content = response_text(data)
        suffix = f":{reasoning_effort}" if reasoning_effort else ""
        return content.strip() if content else None, f"openai_responses:{model}{suffix}"
    except Exception as exc:  # pragma: no cover - provider/network dependent
        return None, f"openai_responses_error:{provider_error_summary(exc)}"


def response_text(data: dict[str, Any]) -> str:
    direct = data.get("output_text")
    if isinstance(direct, str) and direct.strip():
        return direct

    parts: list[str] = []
    output = data.get("output", [])
    if isinstance(output, list):
        for item in output:
            if not isinstance(item, dict):
                continue
            content = item.get("content", [])
            if isinstance(content, list):
                for block in content:
                    if not isinstance(block, dict):
                        continue
                    text = block.get("text")
                    if isinstance(text, str):
                        parts.append(text)
            elif isinstance(content, str):
                parts.append(content)
    return "\n".join(part for part in parts if part.strip())


def provider_error_summary(exc: Exception) -> str:
    response = getattr(exc, "response", None)
    if response is None:
        return str(exc)
    status = getattr(response, "status_code", "unknown")
    try:
        payload = response.json()
    except Exception:
        return f"HTTP {status}"
    error = payload.get("error", payload) if isinstance(payload, dict) else {}
    if isinstance(error, dict):
        code = error.get("code") or error.get("type") or "unknown"
        message = str(error.get("message", "")).strip()
        if message:
            return f"HTTP {status} {code}: {message}"
        return f"HTTP {status} {code}"
    return f"HTTP {status}"


def fallback_report(context: dict[str, Any], mode: str, provider_status: str) -> str:
    if mode == "after_close":
        return fallback_after_close_report(context, provider_status)

    as_of = str(context.get("as_of", "n/a"))
    market_state = context.get("market_state", {})
    portfolio = context.get("portfolio", {})
    holdings = context.get("holdings_review", [])
    candidates = context.get("top_candidates", [])
    options = context.get("options_review", {})
    watchlist = context.get("watchlist_review", {})
    warnings = context.get("warnings", [])
    cash_value = _float_or_none(portfolio.get("cash_value"))
    negative_cash = cash_value is not None and cash_value < 0
    margin_status = controlled_margin_status(market_state, portfolio)

    category_a = [row for row in candidates if row.get("category") == "A"][:5]
    wait_list = [row for row in candidates if row.get("category") in {"B", "D"}][:5]
    no_chase = [row for row in candidates if row.get("category") == "C"][:5]
    reduce_list = [
        row
        for row in holdings
        if _float_or_none(row.get("suggested_sell_pct")) and (_float_or_none(row.get("suggested_sell_pct")) or 0) > 0
    ][:5]

    lines = [
        f"美股盤前快報｜{as_of}",
        f"結論：{premarket_conclusion(negative_cash, margin_status, category_a, market_state)}",
        "",
        "[盤勢圖表]",
        market_chart_line(market_state),
        ma50_chart_line(market_state),
        cash_chart_line(portfolio, market_state, margin_status),
        "",
        "[昨日追蹤清單]",
        *watchlist_review_lines(watchlist),
        "",
        "[持股處理]",
    ]
    if reduce_list:
        lines.append("先處理：" + "；".join(compact_holding_action(row) for row in reduce_list))
    elif holdings:
        lines.append("沒有強制減碼；照停損價管理。")
    else:
        lines.append("尚未提供 holdings.csv，無法做持股優先判斷。")
    if holdings:
        lines.extend(compact_holdings_table(holdings[:8]))

    lines.extend(["", "[今日可買 / 觀察]"])
    if category_a:
        for row in category_a:
            cash_note = ""
            if negative_cash:
                cash_note = "｜受控融資可小筆" if margin_status["can_add_margin"] else "｜現金為負，先觀察"
            lines.append(compact_candidate(row, prefix="可買", suffix=cash_note))
    else:
        lines.append("可買：沒有 A 類。今天不降低標準。")
    if wait_list:
        lines.append("觀察名單：")
        for row in wait_list:
            lines.append(compact_candidate(row, prefix="觀察"))
    if no_chase:
        lines.append("不追高：" + "；".join(f"{row.get('ticker')} > {_price(row.get('no_chase_above'))}" for row in no_chase[:4]))

    lines.extend(["", "[期權確認]"])
    lines.extend(options_summary_lines(options, compact=True))

    lines.extend(["", "[風控與劇本]"])
    lines.append(f"現金 {_money(portfolio.get('cash_value'))}｜現金比 {_pct(portfolio.get('cash_weight'))}｜目標現金 {market_state.get('cash_target', 'n/a')}")
    lines.append(f"新增曝險上限 {_pct(market_state.get('new_exposure_limit'))}｜單檔上限 {_pct(market_state.get('single_name_limit'))}")
    lines.append(margin_policy_line(margin_status))
    lines.append("上漲：不追跳空，等回測守住再加。")
    lines.append("下跌：跌破停損或市場轉黃/橘/紅，取消加碼。")

    if warnings:
        lines.extend(["", "[資料提醒]"])
        lines.extend(f"- {clean_warning(warning)}" for warning in warnings[:4])
    lines.append(f"\n產生方式：{provider_label(provider_status)}")
    return "\n".join(lines)


def fallback_after_close_report(context: dict[str, Any], provider_status: str) -> str:
    as_of = str(context.get("as_of", "n/a"))
    market_state = context.get("market_state", {})
    portfolio = context.get("portfolio", {})
    review = context.get("after_close_review", {})
    news = context.get("news_review", {})
    options = context.get("options_review", {})
    watchlist_created = context.get("watchlist_created", {})
    warnings = context.get("warnings", [])

    missed_buys = review.get("missed_buy_candidates", []) or []
    missed_adds = review.get("missed_add_candidates", []) or []
    missed_sells = review.get("missed_sell_candidates", []) or []
    big_gainers = review.get("big_gainers_1d", []) or []
    big_losers = review.get("big_losers_1d", []) or []
    sector_up = review.get("sector_movers_up", []) or []
    sector_down = review.get("sector_movers_down", []) or []
    news_items = news.get("items", []) if isinstance(news, dict) else []
    source_status = news.get("source_status", {}) if isinstance(news, dict) else {}
    cash_value = _float_or_none(portfolio.get("cash_value"))
    negative_cash = cash_value is not None and cash_value < 0
    margin_status = controlled_margin_status(market_state, portfolio)

    lines = [
        f"美股盤後覆盤｜{as_of}",
        f"結論：{after_close_conclusion(negative_cash, margin_status, missed_sells, missed_buys, missed_adds)}",
        "",
        "[收盤圖表]",
        market_chart_line(market_state),
        ma50_chart_line(market_state),
        cash_chart_line(portfolio, market_state, margin_status),
    ]

    lines.extend(["", "[明日優先順序]"])
    if missed_sells:
        lines.append("1. 先降風險：" + "；".join(compact_holding_action(row) for row in missed_sells[:5]))
    elif negative_cash:
        lines.append("1. 先降融資：現金為負，新增買進先暫停。")
    else:
        lines.append("1. 沒有強制賣出訊號，照停損管理。")
    if missed_buys:
        action_word = "觀察" if negative_cash and not margin_status["can_add_margin"] else "檢討"
        lines.append(f"2. {action_word}錯過買點：" + "；".join(compact_candidate(row, prefix="") for row in missed_buys[:4]))
    else:
        lines.append("2. 沒有明確該買未買名單。")
    if missed_adds and not negative_cash:
        lines.append("3. 可加碼：" + "；".join(compact_holding_action(row) for row in missed_adds[:4]))
    else:
        lines.append("3. 不主動加碼，等更乾淨買點。")

    lines.extend(["", "[產業與異常波動]"])
    if sector_up:
        lines.append("強勢產業：" + "；".join(compact_sector_line(row) for row in sector_up[:3]))
    if sector_down:
        lines.append("弱勢產業：" + "；".join(compact_sector_line(row) for row in sector_down[:3]))
    if big_gainers:
        lines.append("大漲：" + "；".join(_ticker_move(row) for row in big_gainers[:5]))
    if big_losers:
        lines.append("大跌：" + "；".join(_ticker_move(row) for row in big_losers[:5]))
    if not sector_up and not sector_down and not big_gainers and not big_losers:
        lines.append("沒有達到設定門檻的大漲跌或產業輪動。")

    lines.extend(["", "[新聞與消息]"])
    if news_items:
        for item in news_items[:5]:
            lines.append(compact_news_line(item))
    else:
        requested = source_status.get("symbols_requested", 0)
        returned = source_status.get("symbols_returned", 0)
        lines.append(f"免費新聞源沒有回傳可用標題｜returned {returned}/{requested}。")

    lines.extend(["", "[期權檢查]"])
    lines.extend(options_summary_lines(options, compact=True))

    lines.extend(["", "[已建立明日追蹤清單]"])
    lines.extend(watchlist_created_lines(watchlist_created))

    lines.extend(["", "[風控明日劇本]"])
    lines.append(f"現金 {_money(portfolio.get('cash_value'))}｜股票 {_money(portfolio.get('stock_value'))}｜總權益 {_money(portfolio.get('total_equity'))}")
    lines.append(margin_policy_line(margin_status))
    if negative_cash and not margin_status["can_add_margin"]:
        lines.append("明日：先減碼/降融資，再談新買進。")
    elif missed_buys:
        lines.append("明日：不跳空超過不追價，才允許第一筆試單。")
    else:
        lines.append("明日：維持持股，等待更清楚訊號。")

    scope_note = review.get("scope_note")
    if scope_note:
        lines.extend(["", "[範圍提醒]", f"- {clean_scope_note(scope_note)}"])
    if warnings:
        lines.extend(["", "[資料提醒]"])
        lines.extend(f"- {clean_warning(warning)}" for warning in warnings[:4])
    lines.append(f"\n產生方式：{provider_label(provider_status)}")
    return "\n".join(lines)


def _candidate_line(row: dict[str, Any]) -> str:
    return (
        f"{row.get('ticker')} {row.get('name', '')}｜{row.get('category')}｜分數 {_num(row.get('total_score_100'))}"
        f"｜1D {_pct(row.get('return_1d'))}｜5D {_pct(row.get('return_5d'))}"
        f"｜進場 {_price(row.get('entry_low'))}-{_price(row.get('entry_high'))}"
        f"｜不追高於 {_price(row.get('no_chase_above'))}｜{row.get('reason', '')}"
    )


def _holding_line(row: dict[str, Any]) -> str:
    return (
        f"{row.get('ticker')} {row.get('name', '')}｜{row.get('action', '')}"
        f"｜權重 {_pct(row.get('portfolio_weight'))}｜損益 {_pct(row.get('unrealized_pnl_pct'))}"
        f"｜建議賣出 {_pct(row.get('suggested_sell_pct'))}｜停損 {_price(row.get('stop_loss'))}"
        f"｜{row.get('reason', '')}"
    )


def _sector_line(row: dict[str, Any]) -> str:
    return (
        f"{row.get('sector')} 1D {_pct(row.get('avg_return_1d'))}"
        f" / 5D {_pct(row.get('avg_return_5d'))}"
        f"｜代表 {row.get('top_mover', '')} {_pct(row.get('top_mover_return_1d'))}"
    )


def _ticker_move(row: dict[str, Any]) -> str:
    return f"{row.get('ticker')} {_pct(row.get('return_1d'))}｜{row.get('sector', '')}"


def premarket_conclusion(
    negative_cash: bool,
    margin_status: dict[str, Any],
    category_a: list[dict[str, Any]],
    market_state: dict[str, Any],
) -> str:
    if negative_cash and margin_status.get("is_over_limit"):
        return "不開新倉，先把融資降回上限內。"
    if negative_cash and not margin_status.get("can_add_margin"):
        return "現金為負且融資條件不符，今天只觀察。"
    if category_a and market_state.get("risk_light") in {"green", "neutral"}:
        return "可小買 A 類第一筆，但仍以持股風控優先。"
    return "沒有 A 類買點，保留現金。"


def after_close_conclusion(
    negative_cash: bool,
    margin_status: dict[str, Any],
    missed_sells: list[dict[str, Any]],
    missed_buys: list[dict[str, Any]],
    missed_adds: list[dict[str, Any]],
) -> str:
    if missed_sells:
        return "明天先處理該賣/該減碼，買進排後面。"
    if negative_cash and not margin_status.get("can_add_margin"):
        return "重點是降融資，不是增加新部位。"
    if missed_buys or missed_adds:
        return "有追蹤標的，但只能照價位分批，不追高。"
    return "沒有新的強交易理由，維持紀律。"


def market_chart_line(market_state: dict[str, Any]) -> str:
    adv = int(_float_or_none(market_state.get("advancers")) or 0)
    dec = int(_float_or_none(market_state.get("decliners")) or 0)
    total = adv + dec
    ratio = adv / total if total else 0.0
    return (
        f"廣度 {text_bar(ratio)} {_pct(ratio)}"
        f"｜上漲 {adv} / 下跌 {dec}"
        f"｜{market_state.get('market_day_type', 'n/a')} {risk_label(market_state.get('risk_light'))}"
    )


def ma50_chart_line(market_state: dict[str, Any]) -> str:
    ratio = _float_or_none(market_state.get("above_ma50_ratio")) or 0.0
    return f"MA50 {text_bar(ratio)} {_pct(ratio)}｜20日新高 {market_state.get('new_20d_high_count', 0)}"


def cash_chart_line(portfolio: dict[str, Any], market_state: dict[str, Any], margin_status: dict[str, Any]) -> str:
    cash_weight = _float_or_none(portfolio.get("cash_weight"))
    current_margin = _float_or_none(margin_status.get("current_margin_weight")) or 0.0
    max_margin = _float_or_none(margin_status.get("max_negative_cash_weight")) or 0.0
    margin_ratio = current_margin / max_margin if max_margin > 0 else 0.0
    if cash_weight is not None and cash_weight >= 0:
        return f"現金 {_pct(cash_weight)}｜目標 {market_state.get('cash_target', 'n/a')}"
    return (
        f"融資 {text_bar(min(margin_ratio, 1.0))} {_pct(current_margin)} / 上限 {_pct(max_margin)}"
        f"｜{margin_status.get('short_note', '')}"
    )


def compact_holdings_table(holdings: list[dict[str, Any]]) -> list[str]:
    lines = ["代號｜動作｜權重｜損益｜停損"]
    for row in holdings:
        lines.append(
            f"{row.get('ticker')}｜{short_text(row.get('action'), 8)}｜{_pct(row.get('portfolio_weight'))}"
            f"｜{_pct(row.get('unrealized_pnl_pct'))}｜{_price(row.get('stop_loss'))}"
        )
    return lines


def compact_holding_action(row: dict[str, Any]) -> str:
    sell_pct = _float_or_none(row.get("suggested_sell_pct")) or 0.0
    action = str(row.get("action") or "處理")
    action_text = f"{action}{_pct(sell_pct)}" if sell_pct > 0 else action
    return f"{row.get('ticker')} {action_text}｜停損 {_price(row.get('stop_loss'))}"


def compact_candidate(row: dict[str, Any], prefix: str = "觀察", suffix: str = "") -> str:
    label = f"{prefix}｜" if prefix else ""
    return (
        f"{label}{row.get('ticker')} {row.get('category', '')} 分數 {_num(row.get('total_score_100'))}"
        f"｜進場 {_price(row.get('entry_low'))}-{_price(row.get('entry_high'))}"
        f"｜不追 {_price(row.get('no_chase_above'))}{suffix}"
    )


def compact_sector_line(row: dict[str, Any]) -> str:
    sector = short_text(row.get("sector"), 14)
    return f"{sector} 1D {_pct(row.get('avg_return_1d'))}｜代表 {row.get('top_mover', '')}"


def compact_news_line(item: dict[str, Any]) -> str:
    ticker = item.get("ticker") or "n/a"
    publisher = short_text(item.get("publisher"), 12)
    title = short_text(item.get("title"), 72)
    return f"{ticker}｜{publisher}｜{title}"


def watchlist_review_lines(review: dict[str, Any]) -> list[str]:
    if not isinstance(review, dict):
        return ["沒有可用追蹤清單。"]
    summary = review.get("summary", {}) if isinstance(review.get("summary"), dict) else {}
    active_count = int(_float_or_none(summary.get("active_count")) or 0)
    if active_count <= 0:
        return ["沒有昨日延續到今天的追蹤任務。"]

    triggered = review.get("triggered", []) or []
    missed = review.get("missed", []) or []
    waiting = review.get("still_waiting", []) or []
    no_price = review.get("no_price", []) or []
    lines = [
        f"任務 {active_count}｜觸發 {len(triggered)}｜錯過 {len(missed)}｜等待 {len(waiting)}｜無價 {len(no_price)}",
    ]
    if triggered:
        lines.append("已觸發：" + "；".join(watchlist_item_line(row) for row in triggered[:4]))
    if missed:
        lines.append("已錯過：" + "；".join(watchlist_item_line(row) for row in missed[:4]))
    if waiting:
        lines.append("等待：" + "；".join(watchlist_item_line(row) for row in waiting[:4]))
    if no_price:
        lines.append("無價：" + "；".join(str(row.get("ticker", "")) for row in no_price[:8]))
    return lines


def watchlist_created_lines(created: dict[str, Any]) -> list[str]:
    if not isinstance(created, dict):
        return ["沒有建立明日追蹤清單。"]
    items = created.get("items", []) or []
    valid_for = created.get("valid_for", "n/a")
    if not items:
        return [f"{valid_for} 沒有明確追蹤任務。"]
    sell = [row for row in items if row.get("action") == "sell_or_reduce"]
    buy = [row for row in items if row.get("action") in {"buy_watch", "add_watch"}]
    news = [row for row in items if "news" in str(row.get("action", ""))]
    lines = [f"{valid_for} 共 {len(items)} 檔"]
    if sell:
        lines.append("先處理：" + "；".join(watchlist_plan_line(row) for row in sell[:4]))
    if buy:
        lines.append("買/加追蹤：" + "；".join(watchlist_plan_line(row) for row in buy[:4]))
    if news:
        lines.append("消息追蹤：" + "；".join(str(row.get("ticker", "")) for row in news[:6]))
    return lines


def watchlist_item_line(row: dict[str, Any]) -> str:
    ticker = row.get("ticker", "")
    action = watchlist_action_label(row.get("action"))
    price = _price(row.get("current_price"))
    note = short_text(row.get("evaluation_note"), 18)
    return f"{ticker} {action}｜現價 {price}｜{note}"


def watchlist_plan_line(row: dict[str, Any]) -> str:
    ticker = row.get("ticker", "")
    action = watchlist_action_label(row.get("action"))
    low = _price(row.get("trigger_low"))
    high = _price(row.get("trigger_high"))
    stop = _price(row.get("stop_loss"))
    return f"{ticker} {action}｜觸發 {low}-{high}｜停損 {stop}"


def watchlist_action_label(value: Any) -> str:
    labels = {
        "sell_or_reduce": "賣/減碼",
        "add_watch": "加碼",
        "buy_watch": "買進",
        "risk_news_watch": "風險消息",
        "momentum_news_watch": "動能消息",
    }
    return labels.get(str(value), str(value or "追蹤"))


def text_bar(value: Any, width: int = 10) -> str:
    parsed = _float_or_none(value)
    if parsed is None:
        return "░" * width
    bounded = min(max(parsed, 0.0), 1.0)
    filled = int(round(bounded * width))
    return "█" * filled + "░" * (width - filled)


def risk_label(value: Any) -> str:
    labels = {
        "green": "綠燈",
        "neutral": "中性",
        "yellow": "黃燈",
        "orange": "橘燈",
        "red": "紅燈",
    }
    return labels.get(str(value), str(value or "n/a"))


def short_text(value: Any, max_chars: int = 30) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= max_chars:
        return text
    return text[: max(1, max_chars - 1)] + "…"


def clean_warning(value: Any) -> str:
    text = str(value)
    replacements = {
        "Cash balance is negative after margin balance; apply controlled margin policy before any new buy.": "現金為負；新增買進需先符合受控融資規則。",
        "CASH row exists but cash value is zero; fill shares with available USD cash.": "CASH 為 0；請更新可用美元現金。",
    }
    return replacements.get(text, text)


def clean_scope_note(value: Any) -> str:
    text = str(value)
    replacements = {
        "Review is based on configured universe plus current holdings, not every US-listed stock.": "覆盤根據專案 universe 與目前持股；不是即時逐筆新聞或完整 options flow。",
    }
    return replacements.get(text, text)


def provider_label(value: Any) -> str:
    text = str(value)
    if text == "ai_disabled":
        return "本地規則，未呼叫 OpenAI API"
    if text == "non_trading_day":
        return "美股休市檢查"
    return text


def options_summary_lines(options: dict[str, Any], compact: bool = False) -> list[str]:
    if not isinstance(options, dict) or not options.get("enabled"):
        return ["期權檢查未啟用。"]

    bullish = options.get("bullish_confirmations", []) or []
    bearish = options.get("bearish_warnings", []) or []
    high_iv = options.get("high_iv_watchlist", []) or []
    status = options.get("source_status", {}) if isinstance(options.get("source_status"), dict) else {}
    lines: list[str] = []

    if bullish:
        formatter = compact_option_line if compact else _option_line
        lines.append("偏多：" + "；".join(formatter(row) for row in bullish[:4]))
    if bearish:
        formatter = compact_option_line if compact else _option_line
        lines.append("偏空：" + "；".join(formatter(row) for row in bearish[:4]))
    if high_iv:
        lines.append("IV 高：" + "；".join(_option_iv_line(row) for row in high_iv[:4]))
    if not lines:
        requested = status.get("symbols_requested", 0)
        returned = status.get("symbols_returned", 0)
        lines.append(f"未出現明確偏多/偏空｜已查 {returned}/{requested} 檔。")
    note = options.get("scope_note")
    if note:
        lines.append("範圍：非全市場 options flow。")
    return lines


def compact_option_line(row: dict[str, Any]) -> str:
    side = row.get("strongest_contract_side") or "n/a"
    strike = _price(row.get("strongest_contract_strike"))
    return (
        f"{row.get('ticker')} C/P {_num(row.get('call_put_volume_ratio'))}"
        f"｜IV {_pct(row.get('avg_near_atm_iv'))}"
        f"｜最大量 {side} {strike}"
    )


def _option_line(row: dict[str, Any]) -> str:
    side = row.get("strongest_contract_side") or "n/a"
    expiry = row.get("strongest_contract_expiry") or "n/a"
    strike = _price(row.get("strongest_contract_strike"))
    return (
        f"{row.get('ticker')} {row.get('alert', '')}"
        f"｜C/P {_num(row.get('call_put_volume_ratio'))}"
        f"｜近ATM C/P {_num(row.get('near_atm_call_put_ratio'))}"
        f"｜IV {_pct(row.get('avg_near_atm_iv'))}"
        f"｜最大量 {side} {expiry} {strike}"
    )


def _option_iv_line(row: dict[str, Any]) -> str:
    return f"{row.get('ticker')} IV {_pct(row.get('avg_near_atm_iv'))}｜{row.get('alert', '')}"


def controlled_margin_status(market_state: dict[str, Any], portfolio: dict[str, Any]) -> dict[str, Any]:
    policy = dict(market_state.get("margin_policy", {}))
    enabled = bool(policy.get("enabled", market_state.get("margin_allowed", False)))
    policy_allowed = bool(policy.get("allowed", market_state.get("margin_allowed", False)))
    max_negative = _float_or_none(policy.get("max_negative_cash_weight")) or 0.0
    cash_weight = _float_or_none(portfolio.get("cash_weight"))
    current_margin_weight = max(0.0, -(cash_weight or 0.0))
    available_margin_weight = max(0.0, max_negative - current_margin_weight)
    is_over_limit = current_margin_weight >= max_negative if max_negative > 0 else current_margin_weight > 0
    can_add_margin = enabled and policy_allowed and available_margin_weight > 0

    if not enabled:
        short_note = "融資未啟用"
    elif not policy_allowed:
        short_note = "風險燈號不允許融資"
    elif is_over_limit:
        short_note = f"融資已達/超過上限 {_pct(max_negative)}"
    else:
        short_note = f"融資仍有空間 {_pct(available_margin_weight)}"

    return {
        "enabled": enabled,
        "policy_allowed": policy_allowed,
        "can_add_margin": can_add_margin,
        "is_over_limit": is_over_limit,
        "current_margin_weight": current_margin_weight,
        "available_margin_weight": available_margin_weight,
        "max_negative_cash_weight": max_negative,
        "max_single_margin_trade_pct": _float_or_none(policy.get("max_single_margin_trade_pct")) or 0.0,
        "max_new_margin_exposure_pct": _float_or_none(policy.get("max_new_margin_exposure_pct")) or 0.0,
        "allowed_categories": policy.get("allowed_categories", ["A"]),
        "short_note": short_note,
    }


def margin_policy_line(status: dict[str, Any]) -> str:
    if not status.get("enabled"):
        return "融資：未啟用"
    if not status.get("policy_allowed"):
        return f"融資：不可新增（{status.get('short_note')}）"
    if status.get("is_over_limit"):
        return (
            "融資：不可新增"
            f"（目前 {_pct(status.get('current_margin_weight'))} / 上限 {_pct(status.get('max_negative_cash_weight'))}）"
        )
    return (
        "融資：受控可用"
        f"（目前 {_pct(status.get('current_margin_weight'))} / 上限 {_pct(status.get('max_negative_cash_weight'))}"
        f"，單筆融資上限 {_pct(status.get('max_single_margin_trade_pct'))}，只限 A 類）"
    )


def _float_or_none(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def sanitize_for_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): sanitize_for_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [sanitize_for_json(item) for item in value]
    if isinstance(value, tuple):
        return [sanitize_for_json(item) for item in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if hasattr(value, "item"):
        try:
            return sanitize_for_json(value.item())
        except Exception:
            return str(value)
    return value


def _pct(value: Any) -> str:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if not math.isfinite(parsed):
        return "n/a"
    return f"{parsed * 100:.1f}%"


def _num(value: Any) -> str:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if not math.isfinite(parsed):
        return "n/a"
    return f"{parsed:.1f}"


def _price(value: Any) -> str:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if not math.isfinite(parsed):
        return "n/a"
    return f"{parsed:.2f}"


def _money(value: Any) -> str:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if not math.isfinite(parsed):
        return "n/a"
    return f"${parsed:,.0f}"

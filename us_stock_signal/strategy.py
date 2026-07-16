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
    warnings = context.get("warnings", [])
    cash_value = _float_or_none(portfolio.get("cash_value"))
    negative_cash = cash_value is not None and cash_value < 0
    margin_status = controlled_margin_status(market_state, portfolio)
    title = "美股波段盤前策略" if mode == "premarket" else "美股波段盤後覆盤"

    category_a = [row for row in candidates if row.get("category") == "A"][:5]
    wait_list = [row for row in candidates if row.get("category") in {"B", "D"}][:5]
    no_chase = [row for row in candidates if row.get("category") == "C"][:5]

    lines = [
        f"{title}｜{as_of}",
        f"AI 狀態：{provider_status}",
        "",
        "1. 今日盤勢總結",
        f"市場狀態：{market_state.get('market_day_type', 'n/a')}｜風險燈號：{market_state.get('risk_light', 'n/a')}",
        f"廣度：上漲 {market_state.get('advancers', 0)} / 下跌 {market_state.get('decliners', 0)}｜MA50 上方比例 {_pct(market_state.get('above_ma50_ratio'))}",
        "",
        "2. 我的持股操作",
    ]
    if holdings:
        for row in holdings[:8]:
            lines.append(
                f"{row.get('ticker')} {row.get('name')}｜{row.get('action')}｜權重 {_pct(row.get('portfolio_weight'))}｜損益 {_pct(row.get('unrealized_pnl_pct'))}｜停損 {_price(row.get('stop_loss'))}｜{row.get('reason')}"
            )
    else:
        lines.append("尚未提供 holdings.csv；無法做 Portfolio First 的完整持股判斷。")

    lines.extend(
        [
            "",
            "3. 今日可布局股票",
        ]
    )
    if category_a:
        for row in category_a:
            cash_note = ""
            if negative_cash:
                cash_note = "｜可用受控融資小筆" if margin_status["can_add_margin"] else f"｜{margin_status['short_note']}，僅觀察"
            lines.append(
                f"{row.get('ticker')}｜分數 {_num(row.get('total_score_100'))}｜第一筆 {_pct(row.get('first_tranche_pct'))}｜進場 {_price(row.get('entry_low'))}-{_price(row.get('entry_high'))}｜停損 {_price(row.get('stop_loss'))}{cash_note}"
            )
    else:
        lines.append("沒有 A 類候選；今日保留現金，不因有現金而降低標準。")

    lines.extend(["", "4. 等回檔 / 中期觀察"])
    if wait_list:
        for row in wait_list:
            lines.append(f"{row.get('ticker')}｜{row.get('category')}｜分數 {_num(row.get('total_score_100'))}｜等待區 {_price(row.get('entry_low'))}-{_price(row.get('entry_high'))}｜{row.get('reason')}")
    else:
        lines.append("沒有明確等回檔名單。")

    lines.extend(["", "5. 不建議追的股票"])
    if no_chase:
        for row in no_chase:
            lines.append(f"{row.get('ticker')}｜不追高於 {_price(row.get('no_chase_above'))}｜{row.get('reason')}")
    else:
        lines.append("目前沒有因過度延伸被列為 C 類的前排候選。")

    lines.extend(["", "6. 期權確認"])
    lines.extend(options_summary_lines(options))

    lines.extend(
        [
            "",
            "7. 資金配置",
            f"帳戶現金 {_money(portfolio.get('cash_value'))}｜現金比 {_pct(portfolio.get('cash_weight'))}｜策略建議現金 {market_state.get('cash_target', 'n/a')}",
            f"新增曝險上限 {_pct(market_state.get('new_exposure_limit'))}｜單檔上限 {_pct(market_state.get('single_name_limit'))}｜{margin_policy_line(margin_status)}",
            "",
            "8. 明日雙向劇本",
            "上漲：不追超過 no_chase_above 的跳空，等回測或量縮守住再加。",
            "下跌：先分辨正常回檔與失效；跌破停損或市場升級黃/橘/紅燈時取消加碼。",
            "",
            "9. 最終一句話結論",
        ]
    )
    if negative_cash:
        if margin_status["can_add_margin"] and category_a:
            lines.append("帳戶現金為負但仍在受控上限內；只允許 A 類小額融資，跌破停損立即退出。")
        elif margin_status["is_over_limit"]:
            lines.append("融資已達/超過受控上限，不開新倉；先把負現金降回上限內。")
        else:
            lines.append("帳戶現金為負且融資條件不符，不開新倉；先控風險。")
    elif category_a and market_state.get("risk_light") in {"green", "neutral"}:
        lines.append("今天最多小買第一筆，仍以持股風控與保留現金優先。")
    else:
        lines.append("今日沒有比保留現金更好的交易，先保護資金與既有獲利。")
    if warnings:
        lines.extend(["", "資料提醒"])
        lines.extend(f"- {warning}" for warning in warnings[:5])
    return "\n".join(lines)


def fallback_after_close_report(context: dict[str, Any], provider_status: str) -> str:
    as_of = str(context.get("as_of", "n/a"))
    market_state = context.get("market_state", {})
    portfolio = context.get("portfolio", {})
    review = context.get("after_close_review", {})
    news = context.get("news_review", {})
    options = context.get("options_review", {})
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
        f"美股波段盤後覆盤｜{as_of}",
        f"AI 狀態：{provider_status}",
        "",
        "1. 收盤總結",
        f"市場狀態：{market_state.get('market_day_type', 'n/a')}｜風險燈號：{market_state.get('risk_light', 'n/a')}",
        f"廣度：上漲 {market_state.get('advancers', 0)} / 下跌 {market_state.get('decliners', 0)}｜MA50 上方比例 {_pct(market_state.get('above_ma50_ratio'))}",
        f"帳戶現金 {_money(portfolio.get('cash_value'))}｜股票市值 {_money(portfolio.get('stock_value'))}｜總權益 {_money(portfolio.get('total_equity'))}",
    ]
    if negative_cash:
        if margin_status["can_add_margin"]:
            lines.append("現金為負但仍在受控融資上限內：只有 A 類、回測買點、非追高才可討論小額融資。")
        else:
            lines.append(f"現金為負且{margin_status['short_note']}：所有可買/可加先當作觀察。")

    lines.extend(["", "2. 該入手但沒買 / 明天要追蹤"])
    if missed_buys:
        for row in missed_buys[:6]:
            prefix = "可評估" if negative_cash and margin_status["can_add_margin"] else "觀察" if negative_cash else "檢討"
            lines.append(f"{prefix}｜{_candidate_line(row)}")
    else:
        lines.append("沒有明確未持有且達到買進檢討門檻的標的。")

    lines.extend(["", "3. 該加碼但沒加"])
    if missed_adds:
        for row in missed_adds[:6]:
            lines.append(_holding_line(row))
    else:
        lines.append("既有持股沒有明確加碼訊號；若帳戶現金仍為負，這是合理結果。")

    lines.extend(["", "4. 該賣沒賣 / 風險未降"])
    if missed_sells:
        for row in missed_sells[:8]:
            lines.append(_holding_line(row))
    else:
        lines.append("沒有新的強制減碼或停損訊號；仍需照停損價執行。")

    lines.extend(["", "5. 產業與個股異常漲跌"])
    if sector_up:
        lines.append("強勢產業：" + "；".join(_sector_line(row) for row in sector_up[:3]))
    if sector_down:
        lines.append("弱勢產業：" + "；".join(_sector_line(row) for row in sector_down[:3]))
    if big_gainers:
        lines.append("單日大漲：" + "；".join(_ticker_move(row) for row in big_gainers[:5]))
    if big_losers:
        lines.append("單日大跌：" + "；".join(_ticker_move(row) for row in big_losers[:5]))
    if not sector_up and not sector_down and not big_gainers and not big_losers:
        lines.append("沒有達到設定門檻的大漲跌或產業輪動。")

    lines.extend(["", "6. 消息 / 新聞漏看檢查"])
    if news_items:
        for item in news_items[:8]:
            published = item.get("published_at") or "time n/a"
            publisher = item.get("publisher") or "source n/a"
            lines.append(f"{item.get('ticker')}｜{publisher}｜{published}｜{item.get('title')}")
    else:
        requested = source_status.get("symbols_requested", 0)
        returned = source_status.get("symbols_returned", 0)
        lines.append(f"免費新聞源沒有回傳可用標題｜requested {requested} / returned {returned}。重大消息仍需人工看券商/新聞 app。")

    lines.extend(["", "7. 期權檢查"])
    lines.extend(options_summary_lines(options))

    lines.extend(["", "8. 明日行動清單"])
    if missed_sells:
        lines.append("先處理該賣/該減碼清單，再考慮任何新買進。")
    if negative_cash:
        if margin_status["can_add_margin"]:
            lines.append("負現金仍在受控上限內；只允許 A 類小額融資，B 類仍等回檔且不用融資。")
        else:
            lines.append("負現金超過或不符合受控融資條件；候選股只做價格提醒和新聞追蹤。")
    elif missed_buys:
        lines.append("若隔日不跳空超過 no_chase_above，才允許第一筆試單。")
    if missed_adds and not negative_cash:
        lines.append("加碼只在回測支撐不破時做小筆，不追單日長紅。")
    if not missed_sells and not missed_buys and not missed_adds:
        lines.append("明天維持原持股與現金配置，等待更清楚訊號。")

    lines.extend(["", "9. 最終一句話結論"])
    if negative_cash:
        if margin_status["can_add_margin"]:
            lines.append("可保留小額融資彈性，但只能用在 A 類買點，不能用來追 B 類或補弱股。")
        else:
            lines.append("今天的覆盤重點不是多找股票，而是把融資降回受控上限內。")
    elif missed_sells:
        lines.append("明天優先處理風險部位，買進順位排在減碼之後。")
    elif missed_buys or missed_adds:
        lines.append("有可追蹤標的，但只能按價位與倉位規則分批，不追高。")
    else:
        lines.append("盤後沒有足夠的新交易理由，維持紀律比硬交易重要。")

    scope_note = review.get("scope_note")
    if scope_note:
        lines.extend(["", "範圍提醒", f"- {scope_note}"])
    if warnings:
        lines.extend(["", "資料提醒"])
        lines.extend(f"- {warning}" for warning in warnings[:5])
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


def options_summary_lines(options: dict[str, Any]) -> list[str]:
    if not isinstance(options, dict) or not options.get("enabled"):
        return ["期權檢查未啟用。"]

    bullish = options.get("bullish_confirmations", []) or []
    bearish = options.get("bearish_warnings", []) or []
    high_iv = options.get("high_iv_watchlist", []) or []
    status = options.get("source_status", {}) if isinstance(options.get("source_status"), dict) else {}
    lines: list[str] = []

    if bullish:
        lines.append("偏多確認：" + "；".join(_option_line(row) for row in bullish[:4]))
    if bearish:
        lines.append("偏空警訊：" + "；".join(_option_line(row) for row in bearish[:4]))
    if high_iv:
        lines.append("IV 偏高勿追：" + "；".join(_option_iv_line(row) for row in high_iv[:4]))
    if not lines:
        requested = status.get("symbols_requested", 0)
        returned = status.get("symbols_returned", 0)
        lines.append(f"未出現明確期權偏多/偏空確認｜checked {returned}/{requested} 檔。")
    note = options.get("scope_note")
    if note:
        lines.append(f"範圍：{note}")
    return lines


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

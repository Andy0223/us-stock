# US Stock Swing Signal System

這是依 `/Users/andy/tw-stock` 的日常訊號專案風格，重做成美股波段用的精簡版。

核心差異：

- 策略主 prompt 改讀 `/Users/andy/Downloads/美股波段交易完整策略_Prompt.md`。
- Universe 預設改為 Nasdaq Trader symbol directory 產生的美股全市場股票清單，精選池 `data/universe_us_swing.csv` 仍可手動指定。
- 通知 bot 改用獨立環境變數：`US_STOCK_TELEGRAM_BOT_TOKEN` / `US_STOCK_TELEGRAM_CHAT_ID`。
- 流程採 Portfolio First：先讀持股與現金，再看新候選。
- 盤前/盤後會用免費 yfinance option chain 對持股與前排候選做期權確認。
- OpenAI API 是選配；在 Codex 互動使用時可以不打 API，由 Codex 讀 context 後產生完整報告。

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Configure

```bash
cp .env.example .env
cp data/holdings.example.csv data/holdings.csv
```

把 `.env` 填入另一個 Telegram bot：

```bash
US_STOCK_TELEGRAM_BOT_TOKEN=...
US_STOCK_TELEGRAM_CHAT_ID=...

# Optional: only set to 1 if you want the shell script to call OpenAI API by itself.
US_SWING_USE_OPENAI_API=0

# Optional: auto updates only when UNIVERSE_PATH is data/universe_us_all.csv.
US_SWING_UPDATE_UNIVERSE=auto
US_SWING_INCLUDE_ETFS=0

OPENAI_API_KEY=...
OPENAI_MODEL=gpt-5.6
OPENAI_REASONING_EFFORT=xhigh
```

`data/holdings.csv` 的現金列用 `ticker=CASH`、`asset_type=cash`，`shares` 填美元現金金額。

可選：把當天實際交易放到 `data/trades.csv`，盤後覆盤會顯示今日買進/賣出/換倉摘要。格式可參考 `data/trades.example.csv`。

## Margin policy

融資不是絕對禁止，但預設只允許受控使用：

- 只有 `green` 市場才可新增融資
- 只限 A 類標的
- 不追高，只能在進場區或回測買點小筆
- 負現金比上限預設為 `20%`
- 單筆融資上限預設為權益的 `5%`
- B 類只等回檔，不用融資

目前帳戶若已超過負現金上限，報告會顯示「融資不可新增」，但不再把融資視為永久禁止。

## Options review

期權雷達目前使用 `yfinance` 抓 option chain，預設只檢查：

- 目前持股
- 前排 A/B/D/E 候選股
- 盤後覆盤的 news/watch symbols

預設最多 14 檔、每檔近月 3 個到期日，不做全市場逐筆 options flow。報告會標示：

- call/put volume ratio
- near-ATM call/put ratio
- near-ATM IV
- 最大量合約
- 偏多確認、偏空警訊、IV 偏高勿追

這層只作為股票訊號的確認或降級依據，不是內線消息，也不是完整機構大單流。

## Premarket / after-close loop

盤後覆盤會把隔天需要追蹤的標的寫到：

```bash
data/watchlist_next_day.csv
```

內容包含：

- 該賣/該減碼
- 該買但沒買
- 該加碼但沒加
- 大漲/大跌後需要追消息的標的
- 觸發區、停損、不追價

隔天盤前會先讀這份 watchlist，並在 Telegram 的 `[昨日追蹤清單]` 區塊置頂顯示：

- 已觸發
- 已錯過
- 還在等待

盤後再重新檢討並覆寫下一份 watchlist，形成「盤後任務清單 -> 盤前優先檢查 -> 盤後再覆盤」的閉環。

## Run

跟 `tw-stock` 一樣，日常入口是：

```bash
./scripts/run_daily.sh
```

日常入口預設會先更新 `data/universe_us_all.csv`，再跑策略。這個 universe 來自 Nasdaq Trader 的 `nasdaqlisted.txt` / `otherlisted.txt`，並排除測試股、ETF、權證、rights、units、preferred、SPAC / blank-check 等不適合波段推薦的品種。若 `UNIVERSE_PATH` 指到其他檔案，`US_SWING_UPDATE_UNIVERSE=auto` 會自動略過更新，避免覆寫自訂清單。

手動更新全市場 universe：

```bash
./.venv/bin/python -B scripts/update_us_universe.py --output data/universe_us_all.csv
```

若要臨時切回精選池：

```bash
US_SWING_UPDATE_UNIVERSE=0 UNIVERSE_PATH=data/universe_us_swing.csv ./scripts/run_daily.sh
```

若要把 ETF 也納入全市場 universe：

```bash
US_SWING_INCLUDE_ETFS=1 ./.venv/bin/python -B scripts/update_us_universe.py --output data/universe_us_all.csv --include-etfs
```

腳本會自動偵測 `US_STOCK_TELEGRAM_BOT_TOKEN` 與 `US_STOCK_TELEGRAM_CHAT_ID`；兩者都有值才預設發送 Telegram。也可以用 `US_SWING_SEND_TELEGRAM=1` 或 `0` 強制控制。

盤前等同：

```bash
./scripts/run_us_swing_premarket.sh
```

盤後：

```bash
US_SWING_MODE=after_close ./scripts/run_daily.sh
```

或：

```bash
./scripts/run_us_swing_after_close.sh
```

盤後報告會額外檢討：

- 該入手但沒買的候選股
- 該加碼但沒加的既有持股
- 該賣掉或該減碼但沒處理的部位
- 單日/五日大漲跌、20 日新高/新低與產業輪動
- 相關 ticker 的免費新聞提醒
- 持股/候選股的期權偏多、偏空與 IV 過熱提醒

不送 Telegram、也不打 OpenAI：

```bash
US_SWING_SEND_TELEGRAM=0 US_SWING_SKIP_AI=1 ./scripts/run_us_swing_premarket.sh
```

如果要讓 shell script 自己呼叫 OpenAI API，才設定：

```bash
US_SWING_USE_OPENAI_API=1 ./scripts/run_us_swing_premarket.sh
```

手動指定日期：

```bash
python -B -m us_stock_signal \
  --config config/default_config.json \
  --as-of 2026-07-13 \
  --skip-ai \
  --dry-run \
  --force
```

## Firstrade CSV sync

Firstrade 目前按安全路線使用 CSV 匯入，不做非官方登入或下單 API。

把最新 positions CSV 放到：

```bash
data/inbox/firstrade/
```

再跑：

```bash
./scripts/run_firstrade_sync.sh
```

匯入器只有在能辨識 `ticker/symbol`、`shares/quantity`、`avg_cost/cost basis` 欄位時才會更新 `data/holdings.csv`。如果 Firstrade 匯出欄位不同，會停止並列出偵測到的欄位。

## Outputs

輸出在 `outputs/`：

- `us_swing_report_premarket_YYYYMMDD.md`
- `us_swing_report_after_close_YYYYMMDD.md`
- `us_swing_context_premarket_YYYYMMDD.json`
- `us_swing_context_after_close_YYYYMMDD.json`
- `candidate_scores_YYYYMMDD.csv`
- `holdings_review_YYYYMMDD.csv`
- `market_dashboard_YYYYMMDD.csv`
- `sector_scores_YYYYMMDD.csv`
- `options_review_YYYYMMDD.csv`
- `earnings_calendar_YYYYMMDD.csv`
- `closed_loop_radar_YYYYMMDD.csv`

## Earnings calendar

盤前/盤後會自動讀取 AlphaLab 財報行事曆：

- 盤前顯示今日盤前/盤後財報、持股與候選股近期財報。
- 盤後顯示下一個有財報的盤前清單、追蹤標的財報。
- 財報只作為風險與催化提醒；財報前預設不追高，財報後才看價格與量能確認。

## AI-driven closed loop

每日策略現在會走一個輕量閉環：

1. 讀取最近一次 `event_study_factor_lift_*.csv` 的歷史有效因子。
2. 把當天全市場 scan 的候選股套入這些因子，產生 `closed_loop_radar_YYYYMMDD.csv`。
3. 盤前報告顯示「AI 閉環雷達」，只做早期提醒，不跳過現金、融資、停損與不追價規則。
4. 盤後報告顯示「AI 閉環回饋」，檢討 watchlist 觸發/錯過、該買未買、該賣未賣。
5. 盤後會把閉環雷達的趨勢型標的寫入下一個美股交易日的 watchlist。

大型 event study 不會每天跑；需要更新歷史因子時再手動重跑下一節的研究腳本。

## Swing rally event study

歷史波段大漲因子研究不會放進每日 cron。手動執行：

```bash
./.venv/bin/python scripts/run_event_study.py \
  --as-of 2026-07-17 \
  --lookback-days 1600 \
  --max-symbols 1000 \
  --selection-scores outputs/candidate_scores_20260717.csv
```

預設定義為：未來 60 個交易日最大漲幅至少 30%，且相對 SPY 多 15%。輸出：

- `event_study_report_YYYYMMDD.md`
- `event_study_factor_lift_YYYYMMDD.csv`
- `event_study_sector_summary_YYYYMMDD.csv`
- `event_study_regime_summary_YYYYMMDD.csv`
- `event_study_events_YYYYMMDD.csv`
- `event_study_current_factor_watchlist_YYYYMMDD.csv`

## Notes

這個版本預設使用 `data/universe_us_all.csv` 做美股全市場股票掃描。初次全市場跑價量資料會下載數千檔，時間會比精選池長很多；之後有 cache 會快很多。若要快速研究某個主題，可以指定 `UNIVERSE_PATH=data/universe_us_swing.csv`。

Cron 範例在 `scripts/crontab.us-stock.example`。安裝前先把 `/path/to/us-stock` 換成 `/Users/andy/us-stock`。

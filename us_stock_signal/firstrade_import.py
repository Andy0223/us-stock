from __future__ import annotations

import math
import re
import shutil
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from .data import HOLDING_COLUMNS, load_holdings


TICKER_COLUMNS = ["ticker", "symbol", "股票代號", "股票代码", "代號", "证券代码"]
NAME_COLUMNS = ["name", "description", "security", "security description", "股票名稱", "证券名称", "名稱"]
SHARES_COLUMNS = ["shares", "quantity", "qty", "position", "股數", "數量", "持股"]
AVG_COST_COLUMNS = [
    "avg_cost",
    "average cost",
    "avg cost",
    "cost basis/share",
    "cost/share",
    "average price",
    "成本",
    "平均成本",
    "成交均價",
]


@dataclass
class FirstradeImportResult:
    source: Path | None
    output_path: Path
    holdings: pd.DataFrame
    updated_rows: int
    archived_path: Path | None
    warnings: list[str]


def import_firstrade_positions(
    source: str | Path | None = None,
    inbox_dir: str | Path = "data/inbox/firstrade",
    output_path: str | Path = "data/holdings.csv",
    archive_dir: str | Path = "data/raw/firstrade",
    cash: float | None = None,
    allow_missing: bool = False,
    as_of: date | None = None,
) -> FirstradeImportResult:
    as_of = as_of or date.today()
    csv_path = resolve_source(source, inbox_dir)
    output = Path(output_path).expanduser()
    if csv_path is None:
        if allow_missing:
            return FirstradeImportResult(
                source=None,
                output_path=output,
                holdings=pd.DataFrame(columns=HOLDING_COLUMNS),
                updated_rows=0,
                archived_path=None,
                warnings=[f"No Firstrade CSV found in {Path(inbox_dir).expanduser()}."],
            )
        raise FileNotFoundError(f"No Firstrade CSV found in {Path(inbox_dir).expanduser()}")

    parsed = parse_positions_csv(csv_path)
    existing = load_holdings(output) if output.exists() and output.stat().st_size > 0 else pd.DataFrame(columns=HOLDING_COLUMNS)
    cash_rows = build_cash_rows(existing, cash)
    output_frame = pd.concat([parsed, cash_rows], ignore_index=True) if not cash_rows.empty else parsed
    output.parent.mkdir(parents=True, exist_ok=True)
    output_frame.to_csv(output, index=False)
    archived = archive_source(csv_path, archive_dir, as_of)
    return FirstradeImportResult(
        source=csv_path,
        output_path=output,
        holdings=output_frame,
        updated_rows=len(parsed),
        archived_path=archived,
        warnings=[],
    )


def resolve_source(source: str | Path | None, inbox_dir: str | Path) -> Path | None:
    if source:
        candidate = Path(source).expanduser()
        return candidate if candidate.exists() else None
    inbox = Path(inbox_dir).expanduser()
    if not inbox.exists():
        return None
    files = sorted((item for item in inbox.glob("*.csv") if item.is_file()), key=lambda item: item.stat().st_mtime)
    return files[-1] if files else None


def parse_positions_csv(path: Path) -> pd.DataFrame:
    raw = read_csv_flexible(path)
    if raw.empty:
        raise ValueError(f"Firstrade CSV is empty: {path}")
    columns = normalized_columns(raw)
    ticker_col = find_column(columns, TICKER_COLUMNS)
    shares_col = find_column(columns, SHARES_COLUMNS)
    avg_cost_col = find_column(columns, AVG_COST_COLUMNS)
    name_col = find_column(columns, NAME_COLUMNS)
    missing = []
    if ticker_col is None:
        missing.append("ticker/symbol")
    if shares_col is None:
        missing.append("shares/quantity")
    if avg_cost_col is None:
        missing.append("avg_cost/cost basis")
    if missing:
        raise ValueError(
            "Firstrade CSV columns are not recognized. "
            f"Missing {', '.join(missing)}. Detected columns: {list(raw.columns)}"
        )

    rows: list[dict[str, Any]] = []
    for _, row in raw.iterrows():
        values = row.to_dict()
        ticker = str(values.get(ticker_col, "")).strip().upper()
        if not ticker or ticker in {"CASH", "TOTAL"} or ticker.startswith("TOTAL"):
            continue
        shares = parse_number(values.get(shares_col))
        avg_cost = parse_number(values.get(avg_cost_col))
        if shares is None or avg_cost is None or shares == 0:
            continue
        name = str(values.get(name_col, "")).strip() if name_col else ticker
        rows.append(
            {
                "ticker": ticker,
                "name": name or ticker,
                "shares": shares,
                "avg_cost": avg_cost,
                "asset_type": "stock",
                "trade_type": "Firstrade CSV",
                "thesis": "",
            }
        )
    if not rows:
        raise ValueError("Firstrade CSV did not contain any non-zero stock positions.")
    return pd.DataFrame(rows, columns=HOLDING_COLUMNS)


def read_csv_flexible(path: Path) -> pd.DataFrame:
    for encoding in ["utf-8-sig", "utf-8", "big5", "cp950", "latin1"]:
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path)


def normalized_columns(frame: pd.DataFrame) -> dict[str, str]:
    result: dict[str, str] = {}
    for column in frame.columns:
        key = normalize_label(column)
        if key and key not in result:
            result[key] = str(column)
    return result


def find_column(columns: dict[str, str], aliases: Iterable[str]) -> str | None:
    for alias in aliases:
        key = normalize_label(alias)
        if key in columns:
            return columns[key]
    for key, original in columns.items():
        if any(normalize_label(alias) in key for alias in aliases):
            return original
    return None


def normalize_label(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value).strip().lower())


def parse_number(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        parsed = float(value)
        return parsed if math.isfinite(parsed) else None
    text = str(value).strip()
    if not text or text in {"-", "--", "nan", "NaN"}:
        return None
    negative = text.startswith("(") and text.endswith(")")
    cleaned = re.sub(r"[^0-9.\-]", "", text)
    if cleaned in {"", "-", "."}:
        return None
    try:
        parsed = float(cleaned)
    except ValueError:
        return None
    if negative:
        parsed = -abs(parsed)
    return parsed if math.isfinite(parsed) else None


def build_cash_rows(existing: pd.DataFrame, cash: float | None) -> pd.DataFrame:
    if cash is not None:
        return pd.DataFrame(
            [
                {
                    "ticker": "CASH",
                    "name": "USD Cash",
                    "shares": float(cash),
                    "avg_cost": 1.0,
                    "asset_type": "cash",
                    "trade_type": "",
                    "thesis": "",
                }
            ],
            columns=HOLDING_COLUMNS,
        )
    if existing.empty or "is_cash" not in existing:
        return pd.DataFrame(columns=HOLDING_COLUMNS)
    cash_rows = existing[existing["is_cash"]].copy()
    if cash_rows.empty:
        return pd.DataFrame(columns=HOLDING_COLUMNS)
    return cash_rows.loc[:, HOLDING_COLUMNS]


def archive_source(source: Path, archive_dir: str | Path, as_of: date) -> Path:
    target_dir = Path(archive_dir).expanduser()
    target_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%H%M%S")
    target = target_dir / f"firstrade_positions_{as_of.strftime('%Y%m%d')}_{timestamp}_{source.name}"
    shutil.copy2(source, target)
    return target


def format_import_message(result: FirstradeImportResult) -> str:
    if result.source is None:
        warning = result.warnings[0] if result.warnings else "No Firstrade CSV found."
        return f"Firstrade 持股同步：未更新\n{warning}"
    return (
        "Firstrade 持股同步：完成\n"
        f"來源：{result.source}\n"
        f"更新股票列數：{result.updated_rows}\n"
        f"輸出：{result.output_path}"
    )

from __future__ import annotations

import csv
import re
import shutil
import tempfile
import urllib.request
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import pandas as pd


NASDAQ_LISTED_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
OTHER_LISTED_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"

UNIVERSE_COLUMNS = [
    "ticker",
    "name",
    "sector",
    "theme",
    "risk_bucket",
    "fundamental_score",
    "notes",
    "exchange",
    "raw_symbol",
    "asset_type",
]

EXCHANGE_NAMES = {
    "Q": "NASDAQ",
    "G": "NASDAQ Global Market",
    "S": "NASDAQ Capital Market",
    "N": "NYSE",
    "A": "NYSE American",
    "P": "NYSE Arca",
    "Z": "Cboe BZX",
    "V": "IEX",
}

EXCLUDED_NAME_PATTERNS = [
    " warrant",
    " warrants",
    " right",
    " rights",
    " unit",
    " units",
    " preferred",
    " preference",
    " preferred stock",
    " depositary shares",
    " note due",
    " notes due",
    " senior notes",
    " subordinate",
    " debenture",
    " bond",
    " baby bond",
    " subscription",
    "contingent value right",
    " acquisition corp",
    " acquisition corporation",
    " blank check",
    " spac ",
]

EXCLUDED_SYMBOL_SUFFIXES = [
    "W",
    "WS",
    "WT",
    "R",
    "U",
]

EXCLUDED_RAW_SYMBOLS = {
    "CSQR",
    "DPU",
    "STDN",
    # Nasdaq Trader still lists it, but Yahoo has no usable historical prices.
    "SVA",
}


@dataclass
class UniverseUpdateResult:
    output_path: Path
    raw_dir: Path
    row_count: int
    source_counts: dict[str, int]
    skipped_count: int


def update_us_universe(
    output_path: str | Path = "data/universe_us_all.csv",
    raw_dir: str | Path = "data/raw/universe",
    include_etfs: bool = False,
    min_symbols: int = 1000,
) -> UniverseUpdateResult:
    output = Path(output_path).expanduser()
    raw_root = Path(raw_dir).expanduser()
    raw_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    nasdaq_path = raw_root / f"nasdaqlisted_{timestamp}.txt"
    other_path = raw_root / f"otherlisted_{timestamp}.txt"
    download_file(NASDAQ_LISTED_URL, nasdaq_path)
    download_file(OTHER_LISTED_URL, other_path)
    write_latest_copy(nasdaq_path, raw_root / "nasdaqlisted_latest.txt")
    write_latest_copy(other_path, raw_root / "otherlisted_latest.txt")

    rows, skipped = build_universe_rows(nasdaq_path, other_path, include_etfs=include_etfs)
    if len(rows) < int(min_symbols):
        raise RuntimeError(f"Universe update produced only {len(rows)} rows; refusing to overwrite {output}.")

    frame = pd.DataFrame(rows, columns=UNIVERSE_COLUMNS)
    frame = frame.drop_duplicates(subset=["ticker"]).sort_values(["asset_type", "ticker"]).reset_index(drop=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_csv(frame, output)

    source_counts = frame["exchange"].value_counts().to_dict() if not frame.empty else {}
    return UniverseUpdateResult(
        output_path=output,
        raw_dir=raw_root,
        row_count=len(frame),
        source_counts={str(key): int(value) for key, value in source_counts.items()},
        skipped_count=skipped,
    )


def build_universe_rows(nasdaq_path: Path, other_path: Path, include_etfs: bool = False) -> tuple[list[dict[str, object]], int]:
    rows: list[dict[str, object]] = []
    skipped = 0
    for item in read_pipe_file(nasdaq_path):
        if "Symbol" not in item:
            continue
        parsed = parse_listing(
            raw_symbol=item.get("Symbol", ""),
            name=item.get("Security Name", ""),
            exchange=item.get("Market Category", "Q") or "Q",
            is_etf=item.get("ETF", "N") == "Y",
            test_issue=item.get("Test Issue", "N"),
            financial_status=item.get("Financial Status", "N"),
            include_etfs=include_etfs,
            source="nasdaq",
        )
        if parsed is None:
            skipped += 1
        else:
            rows.append(parsed)

    for item in read_pipe_file(other_path):
        if "ACT Symbol" not in item:
            continue
        parsed = parse_listing(
            raw_symbol=item.get("ACT Symbol", ""),
            name=item.get("Security Name", ""),
            exchange=item.get("Exchange", ""),
            is_etf=item.get("ETF", "N") == "Y",
            test_issue=item.get("Test Issue", "N"),
            financial_status="N",
            include_etfs=include_etfs,
            source="other",
        )
        if parsed is None:
            skipped += 1
        else:
            rows.append(parsed)
    return rows, skipped


def parse_listing(
    raw_symbol: str,
    name: str,
    exchange: str,
    is_etf: bool,
    test_issue: str,
    financial_status: str,
    include_etfs: bool,
    source: str,
) -> dict[str, object] | None:
    raw_symbol = str(raw_symbol).strip()
    name = clean_name(name)
    if not raw_symbol or not name:
        return None
    if raw_symbol.upper() in EXCLUDED_RAW_SYMBOLS:
        return None
    if str(test_issue).upper() == "Y":
        return None
    if source == "nasdaq" and str(financial_status).upper() not in {"N", ""}:
        return None
    if is_etf and not include_etfs:
        return None
    if is_excluded_security(raw_symbol, name, is_etf):
        return None

    ticker = yahoo_symbol(raw_symbol)
    if not ticker or not tradable_symbol_shape(ticker):
        return None
    exchange_name = EXCHANGE_NAMES.get(str(exchange).strip().upper(), str(exchange).strip().upper() or "US")
    asset_type = "etf" if is_etf else "stock"
    sector = classify_sector(name, asset_type)
    return {
        "ticker": ticker,
        "name": name,
        "sector": sector,
        "theme": exchange_name,
        "risk_bucket": risk_bucket_for(name, sector, asset_type),
        "fundamental_score": "",
        "notes": f"source=nasdaqtrader; raw_symbol={raw_symbol}; asset_type={asset_type}",
        "exchange": exchange_name,
        "raw_symbol": raw_symbol,
        "asset_type": asset_type,
    }


def read_pipe_file(path: Path) -> Iterable[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="|")
        for row in reader:
            if not row or row.get("Symbol", "").startswith("File Creation Time"):
                continue
            if not row or row.get("ACT Symbol", "").startswith("File Creation Time"):
                continue
            yield {str(key): str(value) for key, value in row.items() if key is not None}


def download_file(url: str, output: Path) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 us-stock-signal/1.0"})
    with urllib.request.urlopen(request, timeout=45) as response, output.open("wb") as handle:
        shutil.copyfileobj(response, handle)


def write_latest_copy(source: Path, target: Path) -> None:
    shutil.copy2(source, target)


def atomic_write_csv(frame: pd.DataFrame, output: Path) -> None:
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="", dir=str(output.parent), delete=False) as handle:
        temp_path = Path(handle.name)
        frame.to_csv(handle, index=False)
    temp_path.replace(output)


def clean_name(value: str) -> str:
    name = re.sub(r"\s+", " ", str(value).strip())
    name = re.sub(r" - Common Stock$", "", name)
    name = re.sub(r" Common Stock$", "", name)
    name = re.sub(r" Ordinary Shares?$", "", name)
    return name.strip()


def yahoo_symbol(raw_symbol: str) -> str:
    return str(raw_symbol).strip().upper().replace("/", "-").replace(".", "-")


def tradable_symbol_shape(ticker: str) -> bool:
    if len(ticker) > 12:
        return False
    return bool(re.fullmatch(r"[A-Z][A-Z0-9.-]*", ticker))


def is_excluded_security(raw_symbol: str, name: str, is_etf: bool) -> bool:
    lowered = f" {name.lower()} "
    if any(pattern in lowered for pattern in EXCLUDED_NAME_PATTERNS):
        return True
    symbol = raw_symbol.upper()
    if not is_etf and symbol.endswith(tuple(EXCLUDED_SYMBOL_SUFFIXES)) and any(
        word in lowered for word in [" acquisition ", " capital ", " holdings "]
    ):
        return True
    return False


def classify_sector(name: str, asset_type: str) -> str:
    lowered = name.lower()
    if asset_type == "etf":
        return "ETF"
    if any(word in lowered for word in ["semiconductor", "micro devices", "silicon", "integrated circuit"]):
        return "Semiconductors and hardware"
    if any(word in lowered for word in ["software", "cloud", "cyber", "data", "digital", "technology", "ai ", "artificial intelligence"]):
        return "Software and technology"
    if any(word in lowered for word in ["therapeutics", "biopharma", "pharmaceutical", "biotech", "medical", "health"]):
        return "Healthcare and biotech"
    if any(word in lowered for word in ["bank", "bancorp", "financial", "insurance", "capital", "asset management"]):
        return "Financials"
    if any(word in lowered for word in ["energy", "oil", "gas", "solar", "power", "utility", "electric"]):
        return "Energy and utilities"
    if any(word in lowered for word in ["industrial", "aerospace", "defense", "manufacturing", "machinery", "construction"]):
        return "Industrials"
    if any(word in lowered for word in ["retail", "restaurant", "consumer", "brands", "apparel", "food", "beverage"]):
        return "Consumer"
    if any(word in lowered for word in ["reit", "realty", "properties", "property", "office", "residential"]):
        return "Real estate"
    if any(word in lowered for word in ["mining", "steel", "gold", "copper", "materials", "chemical"]):
        return "Materials"
    if any(word in lowered for word in ["ads", "adr", "american depositary"]):
        return "ADR and overseas listings"
    return "US common stocks"


def risk_bucket_for(name: str, sector: str, asset_type: str) -> str:
    lowered = name.lower()
    if asset_type == "etf":
        return "etf"
    if "acquisition" in lowered or "blank check" in lowered:
        return "spac"
    if sector in {"Healthcare and biotech", "Software and technology"}:
        return "high_beta"
    if sector in {"Financials", "Energy and utilities", "Consumer"}:
        return "cyclical"
    return "normal"

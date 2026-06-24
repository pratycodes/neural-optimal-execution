"""Download intraday bars from yfinance into the calibration CSV schema."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download yfinance intraday bars for calibration.")
    parser.add_argument("--symbol", required=True, help="Ticker symbol, for example AAPL.")
    parser.add_argument("--interval", default="5m", help="yfinance interval, for example 1m, 5m, 15m, 30m, 60m.")
    parser.add_argument("--period", default="60d", help="yfinance period, for example 5d, 30d, 60d.")
    parser.add_argument("--output", default=None, help="Output CSV path. Defaults to data/raw/<symbol>_<interval>_<period>.csv.")
    parser.add_argument("--prepost", action="store_true", help="Include pre-market and post-market bars.")
    return parser.parse_args()


def resolve_output_path(symbol: str, interval: str, period: str, output: str | None) -> Path:
    if output is not None:
        path = Path(output)
        return path if path.is_absolute() else ROOT / path
    filename = f"{symbol.upper()}_{interval}_{period}.csv"
    return ROOT / "data" / "raw" / filename


def normalize_yfinance_bars(frame: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Convert yfinance output to the empirical calibration CSV schema."""

    if frame.empty:
        raise ValueError(f"No yfinance bars returned for {symbol}.")
    data = frame.copy()
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [str(column[0]) for column in data.columns]
    data = data.reset_index()
    timestamp_col = "Datetime" if "Datetime" in data.columns else "Date"
    rename = {
        timestamp_col: "timestamp",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume",
    }
    data = data.rename(columns=rename)
    required = ["timestamp", "close", "volume"]
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise ValueError(f"yfinance output missing expected columns: {missing}")
    output_columns = ["timestamp", "symbol", "open", "high", "low", "close", "volume"]
    data["symbol"] = symbol.upper()
    for column in output_columns:
        if column not in data.columns:
            data[column] = pd.NA
    data = data[output_columns].dropna(subset=["timestamp", "close", "volume"])
    if data.empty:
        raise ValueError(f"No usable yfinance bars returned for {symbol}.")
    data["timestamp"] = pd.to_datetime(data["timestamp"]).dt.tz_localize(None)
    return data


def download_yfinance_intraday(symbol: str, interval: str, period: str, prepost: bool) -> pd.DataFrame:
    """Download intraday data using yfinance."""

    try:
        import yfinance as yf
    except ImportError as exc:
        raise RuntimeError("yfinance is not installed. Run `python -m pip install yfinance` first.") from exc

    frame = yf.download(
        tickers=symbol,
        interval=interval,
        period=period,
        prepost=prepost,
        auto_adjust=False,
        progress=False,
        threads=False,
    )
    return normalize_yfinance_bars(frame, symbol)


def main() -> int:
    args = parse_args()
    try:
        bars = download_yfinance_intraday(args.symbol, args.interval, args.period, args.prepost)
    except Exception as exc:
        print(f"Failed to download yfinance intraday data: {exc}", file=sys.stderr)
        return 1

    output_path = resolve_output_path(args.symbol, args.interval, args.period, args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    bars.to_csv(output_path, index=False)
    try:
        display = output_path.relative_to(ROOT)
    except ValueError:
        display = output_path
    print(f"Saved {len(bars)} bars to {display}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

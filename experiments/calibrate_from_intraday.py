"""Calibrate simulator profiles from intraday market data."""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
MPL_CACHE_DIR = Path(tempfile.gettempdir()) / "neural_optimal_execution_matplotlib"
XDG_CACHE_DIR = Path(tempfile.gettempdir()) / "neural_optimal_execution_cache"
MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
XDG_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR))
os.environ.setdefault("XDG_CACHE_HOME", str(XDG_CACHE_DIR))

import matplotlib.pyplot as plt

from neural_optimal_execution.data.calibration import (
    CalibrationOrderSizing,
    apply_order_sizing,
    calibrate_intraday_csv,
    calibration_sufficiency_warnings,
    write_calibrated_config,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibrate execution simulator profiles from intraday CSV data.")
    parser.add_argument("--input", required=True, help="Path to intraday CSV.")
    parser.add_argument("--symbol", required=True, help="Symbol to calibrate.")
    parser.add_argument("--bar-minutes", type=int, required=True, help="Bar size in minutes.")
    parser.add_argument("--output-config", required=True, help="Path for calibrated YAML config.")
    parser.add_argument("--output-dir", default="results", help="Directory for calibration tables and figures.")
    parser.add_argument(
        "--target-order-participation",
        type=float,
        default=0.05,
        help="Target parent order as a fraction of expected horizon volume.",
    )
    parser.add_argument(
        "--parent-order-size",
        type=float,
        default=None,
        help="Optional explicit parent order size in shares.",
    )
    parser.add_argument(
        "--max-participation-rate",
        type=float,
        default=0.10,
        help="Maximum allowed participation rate per bucket.",
    )
    parser.add_argument(
        "--auto-resize-order",
        action="store_true",
        help="Reduce oversized parent orders to the safety-capped feasible size.",
    )
    return parser.parse_args()


def resolve_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def display_path(path: Path) -> Path:
    """Return a project-relative path when possible."""

    try:
        return path.relative_to(ROOT)
    except ValueError:
        return path


def missing_input_message(input_path: str | Path) -> str:
    """Return a clear message for missing raw intraday data."""

    return (
        f"Input CSV not found: {input_path}\n"
        "Place your intraday CSV under data/raw/ or pass the correct --input path.\n"
        "For a smoke test with the synthetic fixture, run:\n"
        "python experiments/calibrate_from_intraday.py "
        "--input tests/fixtures/tiny_intraday.csv "
        "--symbol AAPL "
        "--bar-minutes 5 "
        "--output-config configs/calibrated/AAPL_5min.yaml"
    )


def infeasible_order_message(sizing: CalibrationOrderSizing) -> str:
    """Return a clear message for oversized calibrated parent orders."""

    return (
        "Requested calibrated parent order is too large for the participation safety cap.\n"
        f"  total expected horizon volume: {sizing.total_expected_volume_over_horizon:,.2f} shares\n"
        f"  target order participation: {sizing.target_order_participation:.4f}\n"
        f"  max participation rate: {sizing.max_participation_rate:.4f}\n"
        f"  recommended parent order size: {sizing.recommended_parent_order_size:,.2f} shares\n"
        f"  requested/configured parent order size: {sizing.configured_parent_order_size:,.2f} shares\n"
        f"  feasible maximum order size: {sizing.feasible_max_order_size:,.2f} shares\n"
        "Aborting. Lower --parent-order-size or --target-order-participation, "
        "or pass --auto-resize-order to cap the order automatically."
    )


def auto_resized_order_message(sizing: CalibrationOrderSizing) -> str:
    """Return a warning for auto-resized calibrated parent orders."""

    return (
        "WARNING: requested calibrated parent order exceeded the participation safety cap; "
        f"configured parent_order was reduced to {sizing.configured_parent_order_size:,.2f} shares "
        "because --auto-resize-order was passed."
    )


def plot_curve(x_labels: list[str], values, output_path: str | Path, *, ylabel: str, title: str) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(9, 5))
    plt.plot(range(len(values)), values, marker="o")
    plt.xticks(range(len(x_labels)), x_labels, rotation=45, ha="right")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def main() -> int:
    args = parse_args()
    input_path = resolve_path(args.input)
    if not input_path.exists():
        print(missing_input_message(display_path(input_path)), file=sys.stderr)
        return 1
    output_dir = resolve_path(args.output_dir)
    calibration = calibrate_intraday_csv(input_path, args.symbol, args.bar_minutes)
    for warning in calibration_sufficiency_warnings(calibration):
        print(f"WARNING: {warning}", file=sys.stderr)
    sizing = apply_order_sizing(
        calibration,
        target_order_participation=args.target_order_participation,
        max_participation_rate=args.max_participation_rate,
        parent_order_size=args.parent_order_size,
        auto_resize_order=args.auto_resize_order,
    )
    if not sizing.is_participation_feasible:
        print(infeasible_order_message(sizing), file=sys.stderr)
        return 1
    if sizing.was_auto_resized:
        print(auto_resized_order_message(sizing), file=sys.stderr)
    write_calibrated_config(calibration.config, resolve_path(args.output_config))

    table_path = output_dir / "tables" / "empirical_calibration_summary.csv"
    table_path.parent.mkdir(parents=True, exist_ok=True)
    calibration.summary.to_csv(table_path, index=False)

    bucket_labels = calibration.summary["bucket_time"].astype(str).tolist()
    plot_curve(
        bucket_labels,
        calibration.summary["normalized_volume"],
        output_dir / "figures" / "empirical_volume_curve.png",
        ylabel="Normalized average volume",
        title=f"{args.symbol} empirical intraday volume curve",
    )
    plot_curve(
        bucket_labels,
        calibration.summary["realized_volatility"],
        output_dir / "figures" / "empirical_volatility_curve.png",
        ylabel="Realized log-return volatility",
        title=f"{args.symbol} empirical intraday volatility curve",
    )
    if calibration.has_spread:
        plot_curve(
            bucket_labels,
            calibration.summary["avg_spread"],
            output_dir / "figures" / "empirical_spread_curve.png",
            ylabel="Average bid-ask spread",
            title=f"{args.symbol} empirical spread curve",
        )

    print(f"Saved calibrated config to {display_path(resolve_path(args.output_config))}")
    print(f"Saved summary table to {display_path(table_path)}")
    print(f"Saved calibration figures to {display_path(output_dir / 'figures')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

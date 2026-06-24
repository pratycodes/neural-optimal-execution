"""Empirical calibration from intraday bar data."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

REQUIRED_COLUMNS = {"timestamp", "symbol", "close", "volume"}


@dataclass(slots=True)
class EmpiricalCalibration:
    """Calibration outputs derived from one intraday CSV."""

    symbol: str
    bar_minutes: int
    summary: pd.DataFrame
    config: dict[str, Any]
    has_spread: bool


@dataclass(slots=True)
class CalibrationOrderSizing:
    """Order sizing diagnostics for a calibrated execution config."""

    total_expected_volume_over_horizon: float
    target_order_participation: float
    max_participation_rate: float
    recommended_parent_order_size: float
    configured_parent_order_size: float
    feasible_max_order_size: float
    is_participation_feasible: bool
    was_auto_resized: bool


def calibrate_intraday_csv(path: str | Path, symbol: str, bar_minutes: int) -> EmpiricalCalibration:
    """Estimate simulator profiles from intraday bars for one symbol."""

    if bar_minutes <= 0:
        raise ValueError("bar_minutes must be positive.")
    raw = pd.read_csv(path)
    missing = REQUIRED_COLUMNS.difference(raw.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    bars = _prepare_bars(raw, symbol=symbol, bar_minutes=bar_minutes)
    if bars.empty:
        raise ValueError(f"No rows found for symbol {symbol!r}.")

    bars["log_return"] = bars.groupby("session")["close"].transform(lambda values: np.log(values).diff())
    grouped = bars.groupby("bucket_start_min", sort=True)
    avg_volume = grouped["volume"].mean()
    total_avg_volume = float(avg_volume.sum())
    if total_avg_volume <= 0.0:
        raise ValueError("Average volume must be positive.")

    realized_volatility = grouped["log_return"].apply(_root_mean_square).fillna(0.0)
    normalized_volume = avg_volume / total_avg_volume
    labels = _liquidity_regime_labels(avg_volume.to_numpy(dtype=float), realized_volatility.to_numpy(dtype=float))
    summary = pd.DataFrame(
        {
            "bucket_start_min": avg_volume.index.astype(int),
            "bucket_time": [_format_bucket_time(int(value)) for value in avg_volume.index],
            "normalized_volume": normalized_volume.to_numpy(dtype=float),
            "avg_volume": avg_volume.to_numpy(dtype=float),
            "realized_volatility": realized_volatility.to_numpy(dtype=float),
            "liquidity_regime": labels,
        }
    )

    has_spread = {"bid", "ask"}.issubset(bars.columns)
    spread_curve: list[float] | None = None
    if has_spread:
        spread = grouped["spread"].mean().reindex(avg_volume.index).fillna(0.0)
        spread_curve = spread.to_numpy(dtype=float).tolist()
        summary["avg_spread"] = spread_curve

    n_steps = len(summary)
    dt = 1.0 / max(n_steps, 1)
    realized_vol = summary["realized_volatility"].to_numpy(dtype=float)
    simulator_volatility_profile = realized_vol / np.sqrt(dt)
    base_volatility = float(simulator_volatility_profile.mean()) if n_steps else 0.0

    environment: dict[str, Any] = {
        "n_steps": int(n_steps),
        "initial_price": float(bars["close"].iloc[0]),
        "base_daily_volume": total_avg_volume,
        "base_volatility": base_volatility,
        "volume_curve": _round_list(summary["normalized_volume"]),
        "volatility_curve": _round_array(simulator_volatility_profile),
    }
    config: dict[str, Any] = {
        "environment": environment,
        "calibration": {
            "symbol": symbol,
            "bar_minutes": int(bar_minutes),
            "start_timestamp": str(bars["timestamp"].min()),
            "end_timestamp": str(bars["timestamp"].max()),
            "sessions": int(bars["session"].nunique()),
            "normalized_volume_curve": _round_list(summary["normalized_volume"]),
            "average_volume_by_bucket": _round_list(summary["avg_volume"]),
            "realized_volatility_by_bucket": _round_list(summary["realized_volatility"]),
            "liquidity_regime_labels": labels,
        },
    }
    if spread_curve is not None:
        config["calibration"]["spread_curve"] = _round_array(np.asarray(spread_curve, dtype=float))

    return EmpiricalCalibration(symbol=symbol, bar_minutes=bar_minutes, summary=summary, config=config, has_spread=has_spread)


def apply_order_sizing(
    calibration: EmpiricalCalibration,
    *,
    target_order_participation: float,
    max_participation_rate: float,
    parent_order_size: float | None = None,
    auto_resize_order: bool = False,
) -> CalibrationOrderSizing:
    """Attach calibrated order size and participation settings to outputs."""

    if target_order_participation <= 0.0:
        raise ValueError("target_order_participation must be positive.")
    if max_participation_rate <= 0.0:
        raise ValueError("max_participation_rate must be positive.")
    if parent_order_size is not None and parent_order_size <= 0.0:
        raise ValueError("parent_order_size must be positive when provided.")

    total_volume = _total_expected_volume_over_horizon(calibration)
    recommended_order = target_order_participation * total_volume
    requested_order = float(parent_order_size) if parent_order_size is not None else recommended_order
    feasible_max_order = 0.8 * max_participation_rate * total_volume
    feasible = requested_order <= feasible_max_order + 1e-9
    configured_order = min(requested_order, feasible_max_order) if auto_resize_order and not feasible else requested_order
    sizing = CalibrationOrderSizing(
        total_expected_volume_over_horizon=float(total_volume),
        target_order_participation=float(target_order_participation),
        max_participation_rate=float(max_participation_rate),
        recommended_parent_order_size=float(recommended_order),
        configured_parent_order_size=float(configured_order),
        feasible_max_order_size=float(feasible_max_order),
        is_participation_feasible=bool(configured_order <= feasible_max_order + 1e-9),
        was_auto_resized=bool(auto_resize_order and not feasible),
    )
    _write_order_sizing(calibration, sizing)
    return sizing


def write_calibrated_config(config: dict[str, Any], path: str | Path) -> None:
    """Write calibrated YAML config."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)


def calibration_sufficiency_warnings(
    calibration: EmpiricalCalibration,
    *,
    min_buckets: int = 20,
    min_sessions: int = 5,
) -> list[str]:
    """Return warnings for calibration samples that are too small or low-information."""

    warnings: list[str] = []
    n_buckets = len(calibration.summary)
    sessions = int(calibration.config.get("calibration", {}).get("sessions", 0))
    volume_curve = calibration.config.get("environment", {}).get("volume_curve", [])

    if n_buckets < min_buckets:
        warnings.append(
            f"fewer than {min_buckets} time buckets are available ({n_buckets}); "
            "calibrated profiles may be too coarse."
        )
    if sessions < min_sessions:
        warnings.append(
            f"fewer than {min_sessions} trading days are available ({sessions}); "
            "day-to-day liquidity variation is under-sampled."
        )
    if not calibration.has_spread:
        warnings.append("bid/ask spread is missing; no empirical spread curve was calibrated.")
    elif "avg_spread" not in calibration.summary:
        warnings.append("bid/ask spread is missing; no empirical spread curve was calibrated.")
    else:
        spread = calibration.summary["avg_spread"].dropna().to_numpy(dtype=float)
        if spread.size == 0:
            warnings.append("bid/ask spread is missing; no empirical spread curve was calibrated.")
        elif float(np.max(spread) - np.min(spread)) <= 1e-12:
            warnings.append("bid/ask spread is constant; spread diagnostics may not be informative.")
    if len(volume_curve) < min_buckets:
        warnings.append(
            f"volume curve has fewer than {min_buckets} points ({len(volume_curve)}); "
            "execution experiments may collapse to very short horizons."
        )
    return warnings


def _prepare_bars(raw: pd.DataFrame, symbol: str, bar_minutes: int) -> pd.DataFrame:
    data = raw.copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"], errors="coerce")
    data = data[data["symbol"] == symbol].copy()
    data = data.dropna(subset=["timestamp", "close", "volume"])
    data["close"] = pd.to_numeric(data["close"], errors="coerce")
    data["volume"] = pd.to_numeric(data["volume"], errors="coerce")
    data = data.dropna(subset=["close", "volume"])
    data = data[data["volume"] >= 0.0]
    if data.empty:
        return data

    for column in ["bid", "ask", "mid"]:
        if column in data.columns:
            data[column] = pd.to_numeric(data[column], errors="coerce")
    data["session"] = data["timestamp"].dt.date.astype(str)
    minutes = data["timestamp"].dt.hour * 60 + data["timestamp"].dt.minute
    data["bucket_start_min"] = (minutes // bar_minutes) * bar_minutes
    aggregations: dict[str, str] = {
        "timestamp": "min",
        "symbol": "last",
        "close": "last",
        "volume": "sum",
    }
    if {"bid", "ask"}.issubset(data.columns):
        aggregations["bid"] = "mean"
        aggregations["ask"] = "mean"
    if "mid" in data.columns:
        aggregations["mid"] = "last"

    bars = (
        data.sort_values("timestamp")
        .groupby(["session", "bucket_start_min"], as_index=False)
        .agg(aggregations)
        .sort_values(["session", "bucket_start_min"])
        .reset_index(drop=True)
    )
    if {"bid", "ask"}.issubset(bars.columns):
        bars["spread"] = (bars["ask"] - bars["bid"]).clip(lower=0.0)
    return bars


def _total_expected_volume_over_horizon(calibration: EmpiricalCalibration) -> float:
    total = float(calibration.summary["avg_volume"].sum())
    if total <= 0.0:
        raise ValueError("total expected volume over horizon must be positive.")
    return total


def _write_order_sizing(calibration: EmpiricalCalibration, sizing: CalibrationOrderSizing) -> None:
    environment = calibration.config.setdefault("environment", {})
    environment["parent_order"] = float(np.round(sizing.configured_parent_order_size, 6))
    environment["participation_rate"] = float(np.round(sizing.max_participation_rate, 12))

    fields = {
        "total_expected_volume_over_horizon": sizing.total_expected_volume_over_horizon,
        "target_order_participation": sizing.target_order_participation,
        "max_participation_rate": sizing.max_participation_rate,
        "recommended_parent_order_size": sizing.recommended_parent_order_size,
        "configured_parent_order_size": sizing.configured_parent_order_size,
        "feasible_max_order_size": sizing.feasible_max_order_size,
        "is_participation_feasible": sizing.is_participation_feasible,
    }
    calibration.config.setdefault("calibration", {}).update(
        {
            key: (bool(value) if isinstance(value, bool) else float(np.round(value, 12)))
            for key, value in fields.items()
        }
    )
    for key, value in fields.items():
        calibration.summary[key] = value


def _root_mean_square(values: pd.Series) -> float:
    finite = values.dropna().to_numpy(dtype=float)
    if finite.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(finite**2)))


def _liquidity_regime_labels(avg_volume: np.ndarray, realized_volatility: np.ndarray) -> list[str]:
    volume_low, volume_high = np.quantile(avg_volume, [0.30, 0.70])
    volatility_high = np.quantile(realized_volatility, 0.70)
    labels: list[str] = []
    for volume, volatility in zip(avg_volume, realized_volatility):
        if volume >= volume_high and volatility <= volatility_high:
            labels.append("high_liquidity")
        elif volume <= volume_low or volatility >= volatility_high:
            labels.append("stressed_liquidity")
        else:
            labels.append("normal_liquidity")
    return labels


def _format_bucket_time(bucket_start_min: int) -> str:
    hour = bucket_start_min // 60
    minute = bucket_start_min % 60
    return f"{hour:02d}:{minute:02d}"


def _round_list(values: pd.Series) -> list[float]:
    return _round_array(values.to_numpy(dtype=float))


def _round_array(values: np.ndarray) -> list[float]:
    return [float(np.round(value, 12)) for value in values]

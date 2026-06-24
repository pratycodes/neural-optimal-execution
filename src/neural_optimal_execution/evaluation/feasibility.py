"""Feasibility checks for execution experiment configurations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TextIO
import sys

import numpy as np

from neural_optimal_execution.config import ExecutionConfig
from neural_optimal_execution.environment.market_simulator import MarketSimulator


@dataclass(slots=True)
class ParticipationFeasibility:
    """Participation-capacity diagnostics for one execution config."""

    label: str
    parent_order_size: float
    total_expected_volume: float
    max_participation_rate: float
    total_participation_capacity: float
    required_participation_rate: float
    feasible: bool


def participation_feasibility(config: ExecutionConfig, label: str = "config") -> ParticipationFeasibility:
    """Compute whether the order can fit inside expected participation capacity."""

    simulator = MarketSimulator(config)
    expected_volume = config.base_daily_volume * simulator.expected_volume_curve
    total_expected_volume = float(np.sum(expected_volume))
    total_capacity = float(config.participation_rate * total_expected_volume)
    required_rate = (
        float(config.parent_order / total_expected_volume)
        if total_expected_volume > 0.0
        else float("inf")
    )
    feasible = bool(config.parent_order <= total_capacity + 1e-9)
    return ParticipationFeasibility(
        label=label,
        parent_order_size=float(config.parent_order),
        total_expected_volume=total_expected_volume,
        max_participation_rate=float(config.participation_rate),
        total_participation_capacity=total_capacity,
        required_participation_rate=required_rate,
        feasible=feasible,
    )


def format_infeasible_warning(report: ParticipationFeasibility, *, allow_infeasible: bool) -> str:
    """Format a clear warning for an infeasible participation configuration."""

    action = (
        "Continuing because --allow-infeasible was passed."
        if allow_infeasible
        else "Aborting. Pass --allow-infeasible only if terminal-forced liquidation is intentional."
    )
    return "\n".join(
        [
            f"WARNING: Participation capacity is infeasible for {report.label}.",
            f"  feasible maximum order size: {report.total_participation_capacity:,.2f} shares",
            f"  current parent order size: {report.parent_order_size:,.2f} shares",
            f"  configured max participation rate: {report.max_participation_rate:.4f}",
            f"  required participation rate: {report.required_participation_rate:.4f}",
            f"  total expected tradable volume: {report.total_expected_volume:,.2f} shares",
            f"  {action}",
        ]
    )


def ensure_participation_feasible(
    config: ExecutionConfig,
    *,
    label: str = "config",
    allow_infeasible: bool = False,
    stream: TextIO | None = None,
) -> bool:
    """Print diagnostics and return whether the caller should continue."""

    report = participation_feasibility(config, label=label)
    if report.feasible:
        return True
    output = stream if stream is not None else sys.stderr
    print(format_infeasible_warning(report, allow_infeasible=allow_infeasible), file=output)
    return allow_infeasible

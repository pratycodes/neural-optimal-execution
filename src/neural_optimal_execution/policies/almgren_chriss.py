"""Almgren-Chriss style execution baselines."""

from __future__ import annotations

import numpy as np

from neural_optimal_execution.environment.execution_env import ExecutionEnv
from neural_optimal_execution.policies.base import BasePolicy


def compute_ac_schedule(
    inventory: float,
    n_steps: int,
    horizon: float,
    volatility: float,
    temporary_impact: float,
    risk_aversion: float,
) -> np.ndarray:
    """Compute a discrete Almgren-Chriss liquidation schedule.

    This uses the standard hyperbolic-sine shape for the risk-averse liquidation
    trajectory under constant volatility and temporary impact. Very small risk
    aversion falls back to TWAP.
    """

    if n_steps <= 0:
        return np.zeros(0, dtype=float)
    if inventory <= 0:
        return np.zeros(n_steps, dtype=float)
    if risk_aversion <= 0.0 or volatility <= 0.0 or temporary_impact <= 0.0:
        return np.full(n_steps, inventory / n_steps, dtype=float)

    total_time = max(horizon, 1e-12)
    dt = total_time / n_steps
    kappa = float(np.sqrt(max(risk_aversion, 0.0) * volatility**2 / max(temporary_impact, 1e-18)))
    kappa_t = kappa * total_time

    if kappa_t < 1e-8:
        return np.full(n_steps, inventory / n_steps, dtype=float)

    grid = np.arange(n_steps + 1, dtype=float) * dt
    if kappa_t > 50.0:
        # Stable approximation for very front-loaded schedules.
        remaining = inventory * np.exp(-kappa * grid)
        remaining[-1] = 0.0
    else:
        denominator = np.sinh(kappa_t)
        remaining = inventory * np.sinh(kappa * (total_time - grid)) / denominator
        remaining[-1] = 0.0

    trades = remaining[:-1] - remaining[1:]
    trades = np.maximum(trades, 0.0)
    if trades.sum() <= 0.0:
        return np.full(n_steps, inventory / n_steps, dtype=float)
    return trades * inventory / trades.sum()


class AlmgrenChrissPolicy(BasePolicy):
    """Static Almgren-Chriss style schedule using initial parameter estimates."""

    name = "Almgren-Chriss"

    def __init__(self, risk_aversion: float = 5.0e-4):
        self.risk_aversion = risk_aversion
        self.schedule: np.ndarray | None = None

    def reset(self, env: ExecutionEnv) -> None:
        if env.market_path is None:
            raise RuntimeError("Environment has not been reset.")
        volatility = float(np.mean(env.market_path.volatility))
        temp_impact = float(np.mean(env.market_path.temp_impact))
        self.schedule = compute_ac_schedule(
            inventory=env.config.parent_order,
            n_steps=env.config.n_steps,
            horizon=env.config.horizon,
            volatility=volatility,
            temporary_impact=temp_impact,
            risk_aversion=self.risk_aversion,
        )

    def act(self, state: np.ndarray, env: ExecutionEnv) -> float:
        if self.schedule is None:
            self.reset(env)
        assert self.schedule is not None
        return float(self.schedule[min(env.step_index, len(self.schedule) - 1)])


class RecalibratedAlmgrenChrissPolicy(BasePolicy):
    """Recompute the AC schedule from current inventory and market state."""

    name = "Recalibrated AC"

    def __init__(self, risk_aversion: float = 5.0e-4):
        self.risk_aversion = risk_aversion

    def act(self, state: np.ndarray, env: ExecutionEnv) -> float:
        if env.market_path is None:
            raise RuntimeError("Environment has not been reset.")
        remaining_steps = env.config.n_steps - env.step_index
        t = env.step_index
        schedule = compute_ac_schedule(
            inventory=env.inventory,
            n_steps=remaining_steps,
            horizon=env.config.horizon * remaining_steps / env.config.n_steps,
            volatility=float(env.market_path.volatility[t]),
            temporary_impact=float(env.market_path.temp_impact[t]),
            risk_aversion=self.risk_aversion,
        )
        return float(schedule[0]) if len(schedule) else 0.0

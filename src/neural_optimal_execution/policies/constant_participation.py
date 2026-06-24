"""Constant participation-rate execution baseline."""

from __future__ import annotations

import numpy as np

from neural_optimal_execution.environment.execution_env import ExecutionEnv
from neural_optimal_execution.policies.base import BasePolicy


class ConstantParticipationPolicy(BasePolicy):
    """Trade at the parent order's expected-volume participation rate."""

    name = "Constant Participation"

    def __init__(self, target_rate: float | None = None):
        self.target_rate = target_rate

    def act(self, state: np.ndarray, env: ExecutionEnv) -> float:
        if env.market_path is None:
            raise RuntimeError("Environment has not been reset.")
        expected_rate = env.config.parent_order / max(env.config.base_daily_volume, 1e-12)
        target_rate = expected_rate if self.target_rate is None else self.target_rate
        bounded_rate = min(max(float(target_rate), 0.0), env.config.participation_rate)
        return min(env.inventory, bounded_rate * env.market_path.volume[env.step_index])

"""VWAP baseline."""

from __future__ import annotations

import numpy as np

from neural_optimal_execution.environment.execution_env import ExecutionEnv
from neural_optimal_execution.policies.base import BasePolicy


class VWAPPolicy(BasePolicy):
    """Trade in proportion to the expected intraday volume curve."""

    name = "VWAP"

    def act(self, state: np.ndarray, env: ExecutionEnv) -> float:
        if env.market_path is None:
            raise RuntimeError("Environment has not been reset.")
        expected = env.market_path.expected_volume
        remaining_curve = expected[env.step_index :]
        denominator = max(float(remaining_curve.sum()), 1e-12)
        fraction = float(expected[env.step_index] / denominator)
        return env.inventory * fraction

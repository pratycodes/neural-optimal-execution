"""TWAP baseline."""

from __future__ import annotations

import numpy as np

from neural_optimal_execution.environment.execution_env import ExecutionEnv
from neural_optimal_execution.policies.base import BasePolicy


class TWAPPolicy(BasePolicy):
    """Trade equal remaining inventory over the remaining time buckets."""

    name = "TWAP"

    def act(self, state: np.ndarray, env: ExecutionEnv) -> float:
        remaining_steps = max(env.config.n_steps - env.step_index, 1)
        return env.inventory / remaining_steps

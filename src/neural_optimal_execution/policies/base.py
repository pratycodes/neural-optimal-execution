"""Base classes for execution policies."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from neural_optimal_execution.environment.execution_env import EpisodeResult, ExecutionEnv


class BasePolicy(ABC):
    """Interface for deterministic execution policies."""

    name: str = "base"

    def reset(self, env: ExecutionEnv) -> None:
        """Reset policy state at the start of an episode."""

    @abstractmethod
    def act(self, state: np.ndarray, env: ExecutionEnv) -> float:
        """Return desired number of shares to sell."""


def run_policy_episode(policy: BasePolicy, env: ExecutionEnv, seed: int | None = None) -> EpisodeResult:
    """Run one complete episode with a policy."""

    state = env.reset(seed=seed)
    policy.reset(env)
    done = False
    while not done:
        action = policy.act(state, env)
        state, done, _ = env.step(action)
    return env.result()

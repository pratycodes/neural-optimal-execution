"""Neural execution policy."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch import nn

from neural_optimal_execution.environment.execution_env import ExecutionEnv
from neural_optimal_execution.policies.base import BasePolicy


class MLPExecutionPolicy(nn.Module):
    """MLP that outputs a valid trade fraction in [0, 1]."""

    def __init__(self, input_dim: int = 9, hidden_sizes: Iterable[int] = (128, 128, 64)):
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_sizes = tuple(int(size) for size in hidden_sizes)
        layers: list[nn.Module] = []
        previous = self.input_dim
        for size in self.hidden_sizes:
            layers.append(nn.Linear(previous, int(size)))
            layers.append(nn.SiLU())
            previous = int(size)
        layers.append(nn.Linear(previous, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Return fraction of the maximum allowable trade."""

        return torch.sigmoid(self.network(state)).squeeze(-1)

    def save(self, path: str | Path) -> None:
        payload = {
            "state_dict": self.state_dict(),
            "input_dim": self.input_dim,
            "hidden_sizes": self.hidden_sizes,
        }
        torch.save(payload, Path(path))

    @classmethod
    def load(
        cls,
        path: str | Path,
        input_dim: int | None = None,
        hidden_sizes: Iterable[int] | None = None,
    ) -> "MLPExecutionPolicy":
        payload = torch.load(Path(path), map_location="cpu")
        model = cls(
            input_dim=input_dim or int(payload.get("input_dim", 9)),
            hidden_sizes=hidden_sizes or tuple(payload.get("hidden_sizes", (128, 128, 64))),
        )
        model.load_state_dict(payload["state_dict"])
        model.eval()
        return model


class TrainedNeuralPolicy(BasePolicy):
    """Adapter that runs a trained torch policy inside the numpy environment."""

    name = "Neural Policy"

    def __init__(self, model: MLPExecutionPolicy):
        self.model = model.eval()

    @classmethod
    def from_checkpoint(cls, path: str | Path) -> "TrainedNeuralPolicy":
        return cls(MLPExecutionPolicy.load(path))

    def act(self, state: np.ndarray, env: ExecutionEnv) -> float:
        tensor_state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            fraction = float(self.model(tensor_state).item())
        max_trade = env.max_trade_size()
        if env.config.terminal_liquidation:
            return fraction * max_trade
        minimum_trade = expected_completion_floor(env)
        return minimum_trade + fraction * max(0.0, max_trade - minimum_trade)


def expected_completion_floor(env: ExecutionEnv) -> float:
    """Minimum current trade needed to finish against the expected future volume curve."""

    if env.market_path is None:
        raise RuntimeError("Environment has not been reset.")
    future_expected_volume = float(env.market_path.expected_volume[env.step_index + 1 :].sum())
    future_capacity = env.config.participation_rate * future_expected_volume
    required_now = max(0.0, env.inventory - future_capacity)
    return min(env.max_trade_size(), required_now)

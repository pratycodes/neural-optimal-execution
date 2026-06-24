"""Single-order optimal execution environment."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from neural_optimal_execution.config import ExecutionConfig
from neural_optimal_execution.environment.market_simulator import MarketPath, MarketSimulator

FloatArray = NDArray[np.float64]


@dataclass(slots=True)
class EpisodeResult:
    """Container for one completed execution episode."""

    shortfall: float
    normalized_shortfall_bps: float
    terminal_inventory: float
    cash: float
    history: dict[str, FloatArray]


@dataclass(slots=True)
class ExecutionEnv:
    """Environment for liquidating a known parent order.

    The environment models a sell order. A policy chooses how many shares to
    sell at each step. Execution price is reduced by temporary impact and by a
    transient impact state left over from previous trades.
    """

    config: ExecutionConfig
    rng: np.random.Generator = field(default_factory=np.random.default_rng)
    simulator: MarketSimulator = field(init=False)
    market_path: MarketPath | None = field(init=False, default=None)
    step_index: int = field(init=False, default=0)
    price: float = field(init=False, default=0.0)
    inventory: float = field(init=False, default=0.0)
    cash: float = field(init=False, default=0.0)
    transient_impact: float = field(init=False, default=0.0)
    done: bool = field(init=False, default=False)
    history: dict[str, list[float]] = field(init=False, default_factory=dict)

    def __post_init__(self) -> None:
        if self.config.side.lower() != "sell":
            raise NotImplementedError("The starter environment implements sell-side liquidation only.")
        self.simulator = MarketSimulator(self.config)
        self.market_path: MarketPath | None = None
        self.reset()

    @property
    def state_dim(self) -> int:
        return 9

    def reset(self, seed: int | None = None) -> FloatArray:
        """Reset the episode and return the initial state."""

        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.market_path = self.simulator.sample(self.rng)
        cfg = self.config
        self.step_index = 0
        self.price = float(cfg.initial_price)
        self.inventory = float(cfg.parent_order)
        self.cash = 0.0
        self.transient_impact = 0.0
        self.done = False
        self.history: dict[str, list[float]] = {
            "step": [],
            "price": [],
            "execution_price": [],
            "inventory": [],
            "trade_size": [],
            "desired_trade_size": [],
            "cash": [],
            "volume": [],
            "expected_volume": [],
            "volatility": [],
            "temp_impact": [],
            "transient_impact": [],
            "participation_limit": [],
            "participation_violation": [],
            "forced_terminal_liquidation": [],
            "regime": [],
        }
        return self.observe()

    def observe(self) -> FloatArray:
        """Return the normalized state vector observed by policies."""

        self._require_path()
        cfg = self.config
        t = min(self.step_index, cfg.n_steps - 1)
        path = self.market_path
        assert path is not None
        average_bucket_volume = cfg.base_daily_volume / cfg.n_steps
        state = np.array(
            [
                self.inventory / cfg.parent_order,
                (cfg.n_steps - self.step_index) / cfg.n_steps,
                self.step_index / cfg.n_steps,
                self.price / cfg.initial_price - 1.0,
                path.volatility[t] / max(cfg.base_volatility, 1e-12),
                path.volume[t] / max(average_bucket_volume, 1e-12),
                path.expected_volume[t] / max(average_bucket_volume, 1e-12),
                path.temp_impact[t] / max(cfg.temp_impact, 1e-12),
                self.transient_impact / max(cfg.initial_price, 1e-12),
            ],
            dtype=np.float64,
        )
        return state

    def max_trade_size(self) -> float:
        """Maximum trade allowed at the current step before terminal handling."""

        self._require_not_done()
        path = self.market_path
        assert path is not None
        participation_limit = self.config.participation_rate * path.volume[self.step_index]
        return float(min(self.inventory, participation_limit))

    def step(self, action: float) -> tuple[FloatArray, bool, dict[str, Any]]:
        """Execute one step.

        Parameters
        ----------
        action:
            Desired number of shares to sell. The environment clips invalid
            actions when ``config.clip_actions`` is true.
        """

        self._require_not_done()
        path = self.market_path
        assert path is not None
        cfg = self.config
        t = self.step_index
        desired = float(action)
        desired = max(0.0, desired)
        participation_limit = float(cfg.participation_rate * path.volume[t])
        final_step = t == cfg.n_steps - 1
        if cfg.clip_actions:
            normally_allowed_trade = min(desired, self.inventory, participation_limit)
        else:
            normally_allowed_trade = min(desired, self.inventory)

        if cfg.terminal_liquidation and final_step:
            trade_size = self.inventory
            forced_terminal_liquidation = max(0.0, trade_size - normally_allowed_trade)
        elif cfg.clip_actions:
            trade_size = min(desired, self.inventory, participation_limit)
            forced_terminal_liquidation = 0.0
        else:
            trade_size = min(desired, self.inventory)
            forced_terminal_liquidation = 0.0

        violation = max(0.0, trade_size - participation_limit)
        execution_price = self.price - path.temp_impact[t] * trade_size - self.transient_impact
        self.cash += trade_size * execution_price
        self.inventory -= trade_size

        self.history["step"].append(float(t))
        self.history["price"].append(float(self.price))
        self.history["execution_price"].append(float(execution_price))
        self.history["inventory"].append(float(self.inventory))
        self.history["trade_size"].append(float(trade_size))
        self.history["desired_trade_size"].append(float(desired))
        self.history["cash"].append(float(self.cash))
        self.history["volume"].append(float(path.volume[t]))
        self.history["expected_volume"].append(float(path.expected_volume[t]))
        self.history["volatility"].append(float(path.volatility[t]))
        self.history["temp_impact"].append(float(path.temp_impact[t]))
        self.history["transient_impact"].append(float(self.transient_impact))
        self.history["participation_limit"].append(float(participation_limit))
        self.history["participation_violation"].append(float(violation))
        self.history["forced_terminal_liquidation"].append(float(forced_terminal_liquidation))
        self.history["regime"].append(float(path.regimes[t]))

        self.transient_impact = (
            cfg.transient_decay * self.transient_impact
            + cfg.transient_strength * trade_size
            + path.transient_noise[t]
        )
        self.price = max(1e-8, self.price * (1.0 + path.returns[t]) - cfg.permanent_impact * trade_size)

        self.step_index += 1
        self.done = self.step_index >= cfg.n_steps
        observation = self.observe() if not self.done else np.zeros(self.state_dim, dtype=np.float64)
        info = {
            "trade_size": trade_size,
            "execution_price": execution_price,
            "inventory": self.inventory,
            "participation_violation": violation,
            "forced_terminal_liquidation": forced_terminal_liquidation,
        }
        return observation, self.done, info

    def result(self) -> EpisodeResult:
        """Return the final episode result."""

        if not self.done:
            raise RuntimeError("Episode is not complete.")
        cfg = self.config
        benchmark_value = cfg.parent_order * cfg.initial_price
        shortfall = benchmark_value - self.cash
        normalized_shortfall_bps = 10_000.0 * shortfall / benchmark_value
        return EpisodeResult(
            shortfall=float(shortfall),
            normalized_shortfall_bps=float(normalized_shortfall_bps),
            terminal_inventory=float(self.inventory),
            cash=float(self.cash),
            history={key: np.asarray(value, dtype=np.float64) for key, value in self.history.items()},
        )

    def _require_path(self) -> None:
        if self.market_path is None:
            raise RuntimeError("Call reset() before using the environment.")

    def _require_not_done(self) -> None:
        self._require_path()
        if self.done:
            raise RuntimeError("Episode is already complete. Call reset().")

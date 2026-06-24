"""Synthetic market path generator for optimal execution experiments.

The generator is intentionally stylized. It is designed to test execution
policies under stochastic liquidity, stochastic volatility, and regime changes,
not to forecast prices.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from neural_optimal_execution.config import ExecutionConfig

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(slots=True)
class MarketPath:
    """One simulated market path over the execution horizon."""

    expected_volume: FloatArray
    volume: FloatArray
    volatility: FloatArray
    temp_impact: FloatArray
    returns: FloatArray
    transient_noise: FloatArray
    regimes: IntArray


def u_shaped_curve(n_steps: int, strength: float = 1.5) -> FloatArray:
    """Create a normalized U-shaped intraday curve.

    The first and last buckets are larger than the middle bucket. The output
    sums to one.
    """

    if n_steps <= 0:
        raise ValueError("n_steps must be positive.")
    x = np.linspace(0.0, 1.0, n_steps)
    raw = 1.0 + strength * (2.0 * x - 1.0) ** 2
    return raw / raw.sum()


class MarketSimulator:
    """Generate stochastic liquidity and volatility paths."""

    # Regime 0: normal, 1: high liquidity/event, 2: stressed liquidity.
    _transition_matrix = np.array(
        [
            [0.90, 0.07, 0.03],
            [0.15, 0.80, 0.05],
            [0.25, 0.05, 0.70],
        ],
        dtype=float,
    )
    _volume_mult = np.array([1.00, 1.60, 0.45], dtype=float)
    _volatility_mult = np.array([1.00, 1.25, 1.90], dtype=float)
    _impact_mult = np.array([1.00, 0.70, 2.25], dtype=float)

    def __init__(self, config: ExecutionConfig):
        self.config = config
        self.expected_volume_curve = _profile_curve(
            config.volume_curve,
            config.n_steps,
            fallback=u_shaped_curve(config.n_steps),
            normalize_sum=True,
            name="volume_curve",
        )
        self.expected_volatility_curve = _profile_curve(
            config.volatility_curve,
            config.n_steps,
            fallback=u_shaped_curve(config.n_steps, strength=0.75),
            normalize_sum=False,
            name="volatility_curve",
        )

    def sample(self, rng: np.random.Generator) -> MarketPath:
        """Sample one market path."""

        cfg = self.config
        n_steps = cfg.n_steps
        regimes = self._sample_regimes(n_steps, rng)

        expected_volume = cfg.base_daily_volume * self.expected_volume_curve
        volume = expected_volume * self._volume_mult[regimes]
        volume *= rng.lognormal(mean=-0.5 * cfg.volume_noise**2, sigma=cfg.volume_noise, size=n_steps)
        volume = np.maximum(volume, 1.0)

        volatility = cfg.base_volatility * self.expected_volatility_curve / max(self.expected_volatility_curve.mean(), 1e-12)
        volatility = volatility * self._volatility_mult[regimes]
        volatility *= rng.lognormal(
            mean=-0.5 * cfg.volatility_noise**2,
            sigma=cfg.volatility_noise,
            size=n_steps,
        )

        temp_impact = cfg.temp_impact * self._impact_mult[regimes]
        temp_impact *= rng.lognormal(mean=-0.5 * 0.15**2, sigma=0.15, size=n_steps)

        dt = cfg.horizon / cfg.n_steps
        returns = volatility * np.sqrt(dt) * rng.standard_normal(n_steps)
        transient_noise = cfg.transient_noise * rng.standard_normal(n_steps)

        return MarketPath(
            expected_volume=expected_volume.astype(float),
            volume=volume.astype(float),
            volatility=volatility.astype(float),
            temp_impact=temp_impact.astype(float),
            returns=returns.astype(float),
            transient_noise=transient_noise.astype(float),
            regimes=regimes.astype(np.int64),
        )

    def _sample_regimes(self, n_steps: int, rng: np.random.Generator) -> IntArray:
        regimes = np.zeros(n_steps, dtype=np.int64)
        for t in range(1, n_steps):
            regimes[t] = rng.choice(3, p=self._transition_matrix[regimes[t - 1]])
        return regimes


def _profile_curve(
    values: tuple[float, ...] | None,
    n_steps: int,
    *,
    fallback: FloatArray,
    normalize_sum: bool,
    name: str,
) -> FloatArray:
    """Validate an optional empirical simulator profile."""

    if values is None:
        return fallback.astype(float)
    curve = np.asarray(values, dtype=float)
    if curve.shape != (n_steps,):
        raise ValueError(f"{name} length must match n_steps ({n_steps}); got {len(curve)}.")
    if not np.all(np.isfinite(curve)):
        raise ValueError(f"{name} must contain only finite values.")
    if np.any(curve < 0.0):
        raise ValueError(f"{name} cannot contain negative values.")
    if normalize_sum:
        total = float(curve.sum())
        if total <= 0.0:
            raise ValueError(f"{name} must have a positive sum.")
        curve = curve / total
    return curve.astype(float)

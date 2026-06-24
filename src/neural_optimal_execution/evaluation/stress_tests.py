"""Stress-test helpers for policy evaluation."""

from __future__ import annotations

from dataclasses import replace

from neural_optimal_execution.config import ExecutionConfig


def liquidity_drought(config: ExecutionConfig) -> ExecutionConfig:
    """Reduce volume and increase temporary impact."""

    return replace(config, base_daily_volume=config.base_daily_volume * 0.50, temp_impact=config.temp_impact * 1.50)


def volatility_spike(config: ExecutionConfig) -> ExecutionConfig:
    """Double volatility."""

    return replace(config, base_volatility=config.base_volatility * 2.00)


def slow_resilience(config: ExecutionConfig) -> ExecutionConfig:
    """Make transient impact decay more slowly."""

    return replace(config, transient_decay=min(0.98, config.transient_decay + 0.10))

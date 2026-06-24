"""Configuration helpers for the execution research framework."""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class ExecutionConfig:
    """Parameters for one execution simulation."""

    parent_order: float = 100_000.0
    n_steps: int = 40
    horizon: float = 1.0
    initial_price: float = 100.0
    base_volatility: float = 0.02
    volatility_noise: float = 0.20
    base_daily_volume: float = 5_000_000.0
    volume_noise: float = 0.30
    volume_curve: tuple[float, ...] | None = None
    volatility_curve: tuple[float, ...] | None = None
    participation_rate: float = 0.10
    temp_impact: float = 2.0e-7
    permanent_impact: float = 2.0e-8
    transient_decay: float = 0.85
    transient_strength: float = 1.0e-7
    transient_noise: float = 0.0
    terminal_liquidation: bool = True
    clip_actions: bool = True
    side: str = "sell"


@dataclass(slots=True)
class EvaluationConfig:
    """Parameters for evaluating policies."""

    n_episodes: int = 250
    seed: int = 7
    cvar_level: float = 0.95


@dataclass(slots=True)
class TrainingConfig:
    """Parameters for neural policy training."""

    seed: int = 7
    batch_size: int = 512
    epochs: int = 200
    learning_rate: float = 1.0e-3
    hidden_sizes: tuple[int, ...] = (128, 128, 64)
    lambda_var: float = 0.50
    alpha_cvar: float = 0.00
    beta_terminal: float = 10.0
    cvar_level: float = 0.95
    grad_clip_norm: float = 1.0
    device: str = "cpu"


@dataclass(slots=True)
class AlmgrenChrissConfig:
    """Parameters for the Almgren-Chriss style baseline."""

    risk_aversion: float = 5.0e-4


def _filter_dataclass_kwargs(cls: type, values: dict[str, Any]) -> dict[str, Any]:
    valid_names = {field.name for field in fields(cls)}
    return {key: value for key, value in values.items() if key in valid_names}


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML file into a dictionary."""

    with Path(path).open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected a mapping in {path}, got {type(loaded)!r}.")
    return loaded


def load_project_config(path: str | Path) -> tuple[ExecutionConfig, EvaluationConfig, TrainingConfig, AlmgrenChrissConfig]:
    """Load all project configuration sections from YAML."""

    raw = load_yaml(path)
    env_raw = raw.get("environment", {})
    for curve_name in ("volume_curve", "volatility_curve"):
        if curve_name in env_raw and env_raw[curve_name] is not None:
            env_raw = {**env_raw, curve_name: tuple(float(value) for value in env_raw[curve_name])}
    env_cfg = ExecutionConfig(**_filter_dataclass_kwargs(ExecutionConfig, env_raw))
    eval_cfg = EvaluationConfig(**_filter_dataclass_kwargs(EvaluationConfig, raw.get("evaluation", {})))
    train_raw = raw.get("training", {})
    if "hidden_sizes" in train_raw:
        train_raw = {**train_raw, "hidden_sizes": tuple(train_raw["hidden_sizes"])}
    train_cfg = TrainingConfig(**_filter_dataclass_kwargs(TrainingConfig, train_raw))
    ac_cfg = AlmgrenChrissConfig(**_filter_dataclass_kwargs(AlmgrenChrissConfig, raw.get("almgren_chriss", {})))
    return env_cfg, eval_cfg, train_cfg, ac_cfg

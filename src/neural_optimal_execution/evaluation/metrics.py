"""Evaluation metrics for execution policies."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from neural_optimal_execution.config import EvaluationConfig, ExecutionConfig
from neural_optimal_execution.environment.execution_env import EpisodeResult, ExecutionEnv
from neural_optimal_execution.policies.base import BasePolicy, run_policy_episode


@dataclass(slots=True)
class PolicyEvaluation:
    """Results from evaluating a policy over many episodes."""

    policy_name: str
    episodes: list[EpisodeResult]

    @property
    def shortfall_bps(self) -> np.ndarray:
        return np.asarray([episode.normalized_shortfall_bps for episode in self.episodes], dtype=float)

    @property
    def terminal_inventory(self) -> np.ndarray:
        return np.asarray([episode.terminal_inventory for episode in self.episodes], dtype=float)


def cvar(values: np.ndarray, level: float = 0.95) -> float:
    """Compute empirical CVaR of the right tail of a loss distribution."""

    if values.size == 0:
        return float("nan")
    threshold = np.quantile(values, level)
    tail = values[values >= threshold]
    if tail.size == 0:
        return float(threshold)
    return float(tail.mean())


def evaluate_policy(
    policy: BasePolicy,
    env_config: ExecutionConfig,
    eval_config: EvaluationConfig,
) -> PolicyEvaluation:
    """Evaluate a policy over multiple Monte Carlo episodes."""

    episodes: list[EpisodeResult] = []
    for episode_idx in range(eval_config.n_episodes):
        env = ExecutionEnv(env_config)
        seed = eval_config.seed + episode_idx
        result = run_policy_episode(policy, env, seed=seed)
        episodes.append(result)
    return PolicyEvaluation(policy_name=policy.name, episodes=episodes)


def summarize_results(evaluations: list[PolicyEvaluation], cvar_level: float = 0.95) -> pd.DataFrame:
    """Summarize execution-cost distributions and constraint behavior."""

    rows: list[dict[str, float | str]] = []
    for evaluation in evaluations:
        losses = evaluation.shortfall_bps
        terminal_inventory = evaluation.terminal_inventory
        participation_violations = np.asarray(
            [episode.history["participation_violation"].sum() for episode in evaluation.episodes],
            dtype=float,
        )
        forced_terminal_liquidation = np.asarray(
            [episode.history["forced_terminal_liquidation"].sum() for episode in evaluation.episodes],
            dtype=float,
        )
        rows.append(
            {
                "policy": evaluation.policy_name,
                "mean_shortfall_bps": float(losses.mean()),
                "std_shortfall_bps": float(losses.std(ddof=1)) if len(losses) > 1 else 0.0,
                f"cvar_{int(cvar_level * 100)}_bps": cvar(losses, cvar_level),
                "p99_shortfall_bps": float(np.quantile(losses, 0.99)),
                "worst_shortfall_bps": float(losses.max()),
                "completion_rate": float(np.mean(np.isclose(terminal_inventory, 0.0, atol=1e-6))),
                "avg_terminal_inventory": float(terminal_inventory.mean()),
                "avg_participation_violation_shares": float(participation_violations.mean()),
                "forced_terminal_liquidation_shares": float(forced_terminal_liquidation.mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("mean_shortfall_bps").reset_index(drop=True)

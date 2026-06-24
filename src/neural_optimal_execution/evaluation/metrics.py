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


def completion_rate(
    terminal_inventory: np.ndarray,
    parent_order: float,
    tolerance_fraction: float = 1.0e-6,
) -> float:
    """Return the fraction of episodes completed within an economic share tolerance."""

    tolerance_shares = max(1.0e-6, max(float(parent_order), 0.0) * max(float(tolerance_fraction), 0.0))
    return float(np.mean(np.abs(terminal_inventory) <= tolerance_shares))


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


def infer_parent_order(episode: EpisodeResult) -> float:
    """Infer the initial parent order from one completed episode history."""

    traded = episode.history.get("trade_size", np.asarray([], dtype=float))
    return float(np.sum(traded) + episode.terminal_inventory)


def constraint_summary_metrics(
    terminal_inventory: np.ndarray,
    participation_violations: np.ndarray,
    forced_terminal_liquidation: np.ndarray,
    parent_order: float,
) -> dict[str, float]:
    """Summarize share-level and parent-order-fraction constraint metrics."""

    denominator = max(float(parent_order), 1e-12)
    return {
        "avg_terminal_inventory": float(terminal_inventory.mean()),
        "avg_participation_violation_shares": float(participation_violations.mean()),
        "forced_terminal_liquidation_shares": float(forced_terminal_liquidation.mean()),
        "terminal_inventory_fraction": float((terminal_inventory / denominator).mean()),
        "forced_terminal_liquidation_fraction": float((forced_terminal_liquidation / denominator).mean()),
        "participation_violation_fraction_of_parent_order": float((participation_violations / denominator).mean()),
    }


def summarize_results(
    evaluations: list[PolicyEvaluation],
    cvar_level: float = 0.95,
    completion_tolerance_fraction: float = 1.0e-6,
) -> pd.DataFrame:
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
        parent_order = infer_parent_order(evaluation.episodes[0]) if evaluation.episodes else 0.0
        constraint_metrics = constraint_summary_metrics(
            terminal_inventory,
            participation_violations,
            forced_terminal_liquidation,
            parent_order,
        )
        rows.append(
            {
                "policy": evaluation.policy_name,
                "mean_shortfall_bps": float(losses.mean()),
                "std_shortfall_bps": float(losses.std(ddof=1)) if len(losses) > 1 else 0.0,
                f"cvar_{int(cvar_level * 100)}_bps": cvar(losses, cvar_level),
                "p99_shortfall_bps": float(np.quantile(losses, 0.99)),
                "worst_shortfall_bps": float(losses.max()),
                "completion_rate": completion_rate(
                    terminal_inventory,
                    parent_order,
                    completion_tolerance_fraction,
                ),
                **constraint_metrics,
            }
        )
    return pd.DataFrame(rows).sort_values("mean_shortfall_bps").reset_index(drop=True)

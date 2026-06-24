"""Run full-comparison robustness checks across evaluation seeds."""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
MPL_CACHE_DIR = Path(tempfile.gettempdir()) / "neural_optimal_execution_matplotlib"
XDG_CACHE_DIR = Path(tempfile.gettempdir()) / "neural_optimal_execution_cache"
MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
XDG_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR))
os.environ.setdefault("XDG_CACHE_HOME", str(XDG_CACHE_DIR))

import matplotlib.pyplot as plt

from neural_optimal_execution.config import EvaluationConfig, ExecutionConfig, load_project_config
from neural_optimal_execution.environment import ExecutionEnv
from neural_optimal_execution.evaluation.feasibility import ensure_participation_feasible
from neural_optimal_execution.evaluation.metrics import cvar
from neural_optimal_execution.evaluation.outputs import display_path, make_run_output_dirs
from neural_optimal_execution.policies import (
    AlmgrenChrissPolicy,
    BasePolicy,
    RecalibratedAlmgrenChrissPolicy,
    TWAPPolicy,
    TrainedNeuralPolicy,
    VWAPPolicy,
    run_policy_episode,
)

SEEDS = (0, 1, 2, 3, 4)
METRICS = (
    "mean_shortfall_bps",
    "std_shortfall_bps",
    "cvar_95_bps",
    "p99_shortfall_bps",
    "worst_shortfall_bps",
    "completion_rate",
    "avg_terminal_inventory",
    "avg_participation_violation_shares",
    "forced_terminal_liquidation_shares",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate policy robustness across random seeds.")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to YAML config.")
    parser.add_argument("--output-dir", default="results", help="Directory for result artifacts.")
    parser.add_argument("--run-name", default=None, help="Optional run name under results/runs/<run-name>.")
    parser.add_argument("--allow-infeasible", action="store_true", help="Continue even if participation capacity is infeasible.")
    return parser.parse_args()


def build_policies(ac_risk_aversion: float, model_path: Path) -> list[BasePolicy]:
    """Build fresh policy instances for one seed evaluation."""

    policies: list[BasePolicy] = [
        TWAPPolicy(),
        VWAPPolicy(),
        AlmgrenChrissPolicy(risk_aversion=ac_risk_aversion),
        RecalibratedAlmgrenChrissPolicy(risk_aversion=ac_risk_aversion),
    ]
    if model_path.exists():
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning, message="You are using `torch.load`.*")
            policies.append(TrainedNeuralPolicy.from_checkpoint(model_path))
    return policies


def summarize_seed(
    policy_name: str,
    seed: int,
    losses: np.ndarray,
    terminal_inventory: np.ndarray,
    violations: np.ndarray,
    forced_terminal_liquidation: np.ndarray | None = None,
) -> dict[str, float | int | str]:
    """Summarize one policy's evaluation for one seed."""

    forced = forced_terminal_liquidation if forced_terminal_liquidation is not None else np.zeros_like(violations)
    return {
        "seed": seed,
        "policy": policy_name,
        "mean_shortfall_bps": float(losses.mean()),
        "std_shortfall_bps": float(losses.std(ddof=1)) if losses.size > 1 else 0.0,
        "cvar_95_bps": cvar(losses, 0.95),
        "p99_shortfall_bps": float(np.quantile(losses, 0.99)),
        "worst_shortfall_bps": float(losses.max()),
        "completion_rate": float(np.mean(np.isclose(terminal_inventory, 0.0, atol=1e-6))),
        "avg_terminal_inventory": float(terminal_inventory.mean()),
        "avg_participation_violation_shares": float(violations.mean()),
        "forced_terminal_liquidation_shares": float(forced.mean()),
    }


def aggregate_seed_metrics(seed_metrics: pd.DataFrame, policy_order: list[str]) -> pd.DataFrame:
    """Aggregate per-seed metrics into one row per policy."""

    rows: list[dict[str, float | int | str]] = []
    for policy in policy_order:
        policy_metrics = seed_metrics[seed_metrics["policy"] == policy]
        if policy_metrics.empty:
            continue
        row: dict[str, float | int | str] = {
            "policy": policy,
            "n_seeds": int(policy_metrics["seed"].nunique()),
        }
        for metric in METRICS:
            row[f"{metric}_mean"] = float(policy_metrics[metric].mean())
            row[f"{metric}_std"] = float(policy_metrics[metric].std(ddof=1)) if len(policy_metrics) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def plot_metric(summary: pd.DataFrame, metric: str, output_path: str | Path, ylabel: str, title: str) -> None:
    """Save a bar chart with across-seed standard-deviation error bars."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    x = np.arange(len(summary))
    plt.figure(figsize=(9, 5))
    plt.bar(x, summary[f"{metric}_mean"], yerr=summary[f"{metric}_std"], capsize=4)
    plt.xticks(x, summary["policy"], rotation=20, ha="right")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def evaluate_for_seed(
    seed: int,
    policies: list[BasePolicy],
    env_cfg: ExecutionConfig,
    eval_cfg: EvaluationConfig,
) -> list[dict[str, float | int | str]]:
    """Evaluate all policies for one random seed."""

    episode_seeds = np.random.default_rng(seed).integers(
        0,
        np.iinfo(np.int32).max,
        size=eval_cfg.n_episodes,
    )
    rows: list[dict[str, float | int | str]] = []
    for policy in policies:
        print(f"Seed {seed}: evaluating {policy.name}...")
        episodes = [
            run_policy_episode(policy, ExecutionEnv(env_cfg), seed=int(episode_seed))
            for episode_seed in episode_seeds
        ]
        losses = np.asarray([episode.normalized_shortfall_bps for episode in episodes], dtype=float)
        terminal_inventory = np.asarray([episode.terminal_inventory for episode in episodes], dtype=float)
        violations = np.asarray(
            [episode.history["participation_violation"].sum() for episode in episodes],
            dtype=float,
        )
        forced_terminal_liquidation = np.asarray(
            [episode.history["forced_terminal_liquidation"].sum() for episode in episodes],
            dtype=float,
        )
        rows.append(
            summarize_seed(
                policy.name,
                seed,
                losses,
                terminal_inventory,
                violations,
                forced_terminal_liquidation,
            )
        )
    return rows


def main() -> None:
    args = parse_args()
    env_cfg, eval_cfg, _train_cfg, ac_cfg = load_project_config(ROOT / args.config)
    if not ensure_participation_feasible(env_cfg, label=args.config, allow_infeasible=args.allow_infeasible):
        raise SystemExit(1)
    output_dirs = make_run_output_dirs(ROOT / args.output_dir, args.run_name)
    output_dir = output_dirs.root
    model_path = output_dirs.models / "neural_policy.pt"
    if args.run_name:
        print(f"Using run output directory: {display_path(output_dir, ROOT)}")

    if not model_path.exists():
        print(f"Skipping Neural Policy because no saved model exists at {display_path(model_path, ROOT)}.")

    all_rows: list[dict[str, float | int | str]] = []
    policy_order: list[str] | None = None
    for seed in SEEDS:
        policies = build_policies(ac_cfg.risk_aversion, model_path)
        if policy_order is None:
            policy_order = [policy.name for policy in policies]
        all_rows.extend(evaluate_for_seed(seed, policies, env_cfg, eval_cfg))

    seed_metrics = pd.DataFrame(all_rows)
    summary = aggregate_seed_metrics(seed_metrics, policy_order or [])

    raw_table_path = output_dirs.tables / "seed_robustness_raw.csv"
    aggregate_table_path = output_dirs.tables / "seed_robustness_metrics.csv"
    seed_metrics.to_csv(raw_table_path, index=False)
    summary.to_csv(aggregate_table_path, index=False)

    plot_metric(
        summary,
        "cvar_95_bps",
        output_dirs.figures / "seed_robustness_cvar.png",
        "CVaR 95 implementation shortfall (bps)",
        "Seed robustness: CVaR 95",
    )
    plot_metric(
        summary,
        "mean_shortfall_bps",
        output_dirs.figures / "seed_robustness_mean_shortfall.png",
        "Mean implementation shortfall (bps)",
        "Seed robustness: mean shortfall",
    )

    print("\nSeed robustness summary:")
    print(summary.to_string(index=False))
    print(f"\nSaved raw table to {display_path(raw_table_path, ROOT)}")
    print(f"Saved aggregate table to {display_path(aggregate_table_path, ROOT)}")
    print(f"Saved figures to {display_path(output_dirs.figures, ROOT)}")


if __name__ == "__main__":
    main()

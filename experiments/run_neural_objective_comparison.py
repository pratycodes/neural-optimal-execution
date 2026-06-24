"""Compare mean-variance and CVaR-aware neural policies across training seeds."""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from neural_optimal_execution.config import load_project_config
from neural_optimal_execution.evaluation.feasibility import ensure_participation_feasible
from neural_optimal_execution.evaluation.metrics import evaluate_policy, summarize_results
from neural_optimal_execution.evaluation.outputs import display_path, make_run_output_dirs
from neural_optimal_execution.policies import TrainedNeuralPolicy
from neural_optimal_execution.training import train_neural_policy

DEFAULT_TRAINING_SEEDS = (0, 1, 2, 3, 4)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare neural mean-variance and CVaR objectives.")
    parser.add_argument(
        "--mean-variance-config",
        default="configs/objectives/mean_variance.yaml",
        help="Mean-variance policy config.",
    )
    parser.add_argument(
        "--cvar-config",
        default="configs/objectives/cvar.yaml",
        help="CVaR-aware policy config.",
    )
    parser.add_argument("--output-dir", default="results", help="Directory for result artifacts.")
    parser.add_argument("--run-name", default="neural_objective_comparison", help="Run name under results/runs/.")
    parser.add_argument(
        "--training-seeds",
        type=int,
        nargs="+",
        default=list(DEFAULT_TRAINING_SEEDS),
        help="Training seeds used for both objectives.",
    )
    return parser.parse_args()


def validate_comparable_configs(mean_variance_config, cvar_config, mean_eval, cvar_eval) -> None:
    """Require both objectives to use identical environment and evaluation settings."""

    if mean_variance_config != cvar_config:
        raise ValueError("Objective comparison configs must have identical environment sections.")
    if mean_eval != cvar_eval:
        raise ValueError("Objective comparison configs must have identical evaluation sections.")


def aggregate_objective_metrics(raw: pd.DataFrame) -> pd.DataFrame:
    """Aggregate each objective's metrics across neural-policy training seeds."""

    numeric_columns = [
        column
        for column in raw.columns
        if column not in {"training_seed", "objective", "policy"}
    ]
    aggregate = raw.groupby(["objective", "policy"])[numeric_columns].agg(["mean", "std"])
    aggregate.columns = [f"{metric}_{statistic}" for metric, statistic in aggregate.columns]
    return aggregate.reset_index()


def main() -> None:
    args = parse_args()
    mean_env, mean_eval, mean_train, _mean_ac = load_project_config(ROOT / args.mean_variance_config)
    cvar_env, cvar_eval, cvar_train, _cvar_ac = load_project_config(ROOT / args.cvar_config)
    validate_comparable_configs(mean_env, cvar_env, mean_eval, cvar_eval)
    if not ensure_participation_feasible(mean_env, label=args.mean_variance_config):
        raise SystemExit(1)

    output_dirs = make_run_output_dirs(ROOT / args.output_dir, args.run_name)
    rows: list[pd.DataFrame] = []
    objectives = (
        ("mean_variance", "Neural Mean-Variance", mean_train),
        ("cvar", "Neural CVaR", cvar_train),
    )
    for training_seed in args.training_seeds:
        for objective_key, policy_name, train_config in objectives:
            seeded_train_config = replace(train_config, seed=training_seed)
            model_path = output_dirs.models / f"{objective_key}_seed_{training_seed}.pt"
            log_path = output_dirs.tables / f"{objective_key}_training_seed_{training_seed}.csv"
            model, _log = train_neural_policy(
                env_config=mean_env,
                train_config=seeded_train_config,
                output_model_path=model_path,
                output_log_path=log_path,
            )
            policy = TrainedNeuralPolicy(model)
            policy.name = policy_name
            evaluation = evaluate_policy(policy, mean_env, mean_eval)
            summary = summarize_results(
                [evaluation],
                cvar_level=mean_eval.cvar_level,
                completion_tolerance_fraction=mean_eval.completion_tolerance_fraction,
            )
            summary.insert(0, "training_seed", training_seed)
            summary.insert(1, "objective", objective_key)
            rows.append(summary)

    raw = pd.concat(rows, ignore_index=True)
    aggregate = aggregate_objective_metrics(raw)
    raw_path = output_dirs.tables / "neural_objective_comparison_raw.csv"
    aggregate_path = output_dirs.tables / "neural_objective_comparison_summary.csv"
    raw.to_csv(raw_path, index=False)
    aggregate.to_csv(aggregate_path, index=False)
    print(aggregate.to_string(index=False))
    print(f"\nSaved raw comparison to {display_path(raw_path, ROOT)}")
    print(f"Saved summary to {display_path(aggregate_path, ROOT)}")


if __name__ == "__main__":
    main()

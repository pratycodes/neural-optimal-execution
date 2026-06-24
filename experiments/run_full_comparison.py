"""Compare classical baselines against a trained neural policy."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from neural_optimal_execution.config import load_project_config
from neural_optimal_execution.evaluation.feasibility import ensure_participation_feasible
from neural_optimal_execution.evaluation.metrics import evaluate_policy, summarize_results
from neural_optimal_execution.evaluation.outputs import display_path, make_run_output_dirs
from neural_optimal_execution.evaluation.plots import plot_average_inventory, plot_cost_distributions
from neural_optimal_execution.policies import (
    AlmgrenChrissPolicy,
    RecalibratedAlmgrenChrissPolicy,
    TWAPPolicy,
    TrainedNeuralPolicy,
    VWAPPolicy,
)
from neural_optimal_execution.training import train_neural_policy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate baselines and neural execution policy.")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to YAML config.")
    parser.add_argument("--output-dir", default="results", help="Directory for result artifacts.")
    parser.add_argument("--run-name", default=None, help="Optional run name under results/runs/<run-name>.")
    parser.add_argument("--retrain", action="store_true", help="Retrain the neural model before evaluation.")
    parser.add_argument("--allow-infeasible", action="store_true", help="Continue even if participation capacity is infeasible.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    env_cfg, eval_cfg, train_cfg, ac_cfg = load_project_config(ROOT / args.config)
    if not ensure_participation_feasible(env_cfg, label=args.config, allow_infeasible=args.allow_infeasible):
        raise SystemExit(1)
    output_dirs = make_run_output_dirs(ROOT / args.output_dir, args.run_name)
    output_dir = output_dirs.root
    model_path = output_dirs.models / "neural_policy.pt"
    log_path = output_dirs.tables / "neural_training_log.csv"
    if args.run_name:
        print(f"Using run output directory: {display_path(output_dir, ROOT)}")

    if args.retrain or not model_path.exists():
        print("Training neural policy...")
        train_neural_policy(
            env_config=env_cfg,
            train_config=train_cfg,
            output_model_path=model_path,
            output_log_path=log_path,
        )

    policies = [
        TWAPPolicy(),
        VWAPPolicy(),
        AlmgrenChrissPolicy(risk_aversion=ac_cfg.risk_aversion),
        RecalibratedAlmgrenChrissPolicy(risk_aversion=ac_cfg.risk_aversion),
        TrainedNeuralPolicy.from_checkpoint(model_path),
    ]
    evaluations = []
    for policy in policies:
        print(f"Evaluating {policy.name}...")
        evaluations.append(evaluate_policy(policy, env_cfg, eval_cfg))

    summary = summarize_results(evaluations, cvar_level=eval_cfg.cvar_level)
    table_path = output_dirs.tables / "full_comparison_metrics.csv"
    summary.to_csv(table_path, index=False)
    plot_cost_distributions(evaluations, output_dirs.figures / "full_cost_distributions.png")
    plot_average_inventory(evaluations, output_dirs.figures / "full_average_inventory.png")
    print("\nPolicy summary:")
    print(summary.to_string(index=False))
    print(f"\nSaved table to {display_path(table_path, ROOT)}")


if __name__ == "__main__":
    main()

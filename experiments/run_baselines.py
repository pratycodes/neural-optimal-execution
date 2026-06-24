"""Run classical execution baselines and save summary artifacts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from neural_optimal_execution.config import load_project_config
from neural_optimal_execution.evaluation.metrics import evaluate_policy, summarize_results
from neural_optimal_execution.evaluation.plots import plot_average_inventory, plot_cost_distributions
from neural_optimal_execution.policies import (
    AlmgrenChrissPolicy,
    ConstantParticipationPolicy,
    RecalibratedAlmgrenChrissPolicy,
    TWAPPolicy,
    VWAPPolicy,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate TWAP, VWAP, and Almgren-Chriss baselines.")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to YAML config.")
    parser.add_argument("--output-dir", default="results", help="Directory for result artifacts.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    env_cfg, eval_cfg, _train_cfg, ac_cfg = load_project_config(ROOT / args.config)
    output_dir = ROOT / args.output_dir
    policies = [
        TWAPPolicy(),
        VWAPPolicy(),
        ConstantParticipationPolicy(),
        AlmgrenChrissPolicy(risk_aversion=ac_cfg.risk_aversion),
        RecalibratedAlmgrenChrissPolicy(risk_aversion=ac_cfg.risk_aversion),
    ]
    evaluations = []
    for policy in policies:
        print(f"Evaluating {policy.name}...")
        evaluations.append(evaluate_policy(policy, env_cfg, eval_cfg))
    summary = summarize_results(
        evaluations,
        cvar_level=eval_cfg.cvar_level,
        completion_tolerance_fraction=eval_cfg.completion_tolerance_fraction,
    )
    table_path = output_dir / "tables" / "baseline_metrics.csv"
    table_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(table_path, index=False)
    plot_cost_distributions(evaluations, output_dir / "figures" / "cost_distributions.png")
    plot_average_inventory(evaluations, output_dir / "figures" / "average_inventory.png")
    print("\nPolicy summary:")
    print(summary.to_string(index=False))
    print(f"\nSaved table to {table_path.relative_to(ROOT)}")
    print(f"Saved figures to {(output_dir / 'figures').relative_to(ROOT)}")


if __name__ == "__main__":
    main()

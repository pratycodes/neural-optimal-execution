"""Train the neural execution policy through differentiable simulation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from neural_optimal_execution.config import load_project_config
from neural_optimal_execution.evaluation.feasibility import ensure_participation_feasible
from neural_optimal_execution.evaluation.outputs import display_path, make_run_output_dirs
from neural_optimal_execution.training import train_neural_policy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train neural execution policy.")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to YAML config.")
    parser.add_argument("--output-dir", default="results", help="Directory for result artifacts.")
    parser.add_argument("--run-name", default=None, help="Optional run name under results/runs/<run-name>.")
    parser.add_argument("--allow-infeasible", action="store_true", help="Continue even if participation capacity is infeasible.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    env_cfg, _eval_cfg, train_cfg, _ac_cfg = load_project_config(ROOT / args.config)
    if not ensure_participation_feasible(env_cfg, label=args.config, allow_infeasible=args.allow_infeasible):
        raise SystemExit(1)
    output_dirs = make_run_output_dirs(ROOT / args.output_dir, args.run_name)
    model_path = output_dirs.models / "neural_policy.pt"
    log_path = output_dirs.tables / "neural_training_log.csv"
    if args.run_name:
        print(f"Using run output directory: {display_path(output_dirs.root, ROOT)}")
    _policy, log_df = train_neural_policy(
        env_config=env_cfg,
        train_config=train_cfg,
        output_model_path=model_path,
        output_log_path=log_path,
    )
    print(log_df.tail().to_string(index=False))
    print(f"\nSaved model to {display_path(model_path, ROOT)}")
    print(f"Saved training log to {display_path(log_path, ROOT)}")


if __name__ == "__main__":
    main()

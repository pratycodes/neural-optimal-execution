"""Measure learned policy sensitivity to each observed state feature."""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from neural_optimal_execution.config import load_project_config
from neural_optimal_execution.environment import ExecutionEnv
from neural_optimal_execution.evaluation.outputs import display_path, make_run_output_dirs
from neural_optimal_execution.evaluation.policy_diagnostics import state_ablation_sensitivity
from neural_optimal_execution.policies import TrainedNeuralPolicy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run neural-policy state ablation diagnostics.")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to YAML config.")
    parser.add_argument("--output-dir", default="results", help="Directory for result artifacts.")
    parser.add_argument("--run-name", default=None, help="Optional run name under results/runs/<run-name>.")
    parser.add_argument("--episodes", type=int, default=20, help="Number of on-policy episodes used to collect states.")
    parser.add_argument("--strict-constraints", action="store_true", help="Use strict participation constraints.")
    return parser.parse_args()


def collect_on_policy_states(
    policy: TrainedNeuralPolicy,
    env_config,
    *,
    episodes: int,
    seed: int,
) -> np.ndarray:
    """Collect states visited by the saved policy without exposing future path values."""

    collected: list[np.ndarray] = []
    for episode_index in range(episodes):
        env = ExecutionEnv(env_config)
        state = env.reset(seed=seed + episode_index)
        done = False
        while not done:
            collected.append(state.copy())
            action = policy.act(state, env)
            state, done, _info = env.step(action)
    return np.asarray(collected, dtype=np.float32)


def main() -> None:
    args = parse_args()
    if args.episodes <= 0:
        raise SystemExit("--episodes must be positive.")
    env_config, eval_config, _train_config, _ac_config = load_project_config(ROOT / args.config)
    if args.strict_constraints:
        env_config = replace(env_config, terminal_liquidation=False, clip_actions=True)
    output_dirs = make_run_output_dirs(ROOT / args.output_dir, args.run_name)
    model_path = output_dirs.models / "neural_policy.pt"
    if not model_path.exists():
        raise SystemExit(f"Saved neural policy not found at {display_path(model_path, ROOT)}.")

    policy = TrainedNeuralPolicy.from_checkpoint(model_path)
    states = collect_on_policy_states(
        policy,
        env_config,
        episodes=args.episodes,
        seed=eval_config.seed,
    )
    diagnostics = state_ablation_sensitivity(policy.model, states)
    output_path = output_dirs.tables / "policy_state_ablation.csv"
    diagnostics.to_csv(output_path, index=False)
    print(diagnostics.to_string(index=False))
    print(f"\nSaved state-ablation diagnostics to {display_path(output_path, ROOT)}")


if __name__ == "__main__":
    main()

"""Plotting helpers for experiment outputs."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from neural_optimal_execution.evaluation.metrics import PolicyEvaluation


def plot_cost_distributions(evaluations: list[PolicyEvaluation], output_path: str | Path) -> None:
    """Save a histogram of implementation shortfall distributions."""

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(9, 5))
    for evaluation in evaluations:
        plt.hist(evaluation.shortfall_bps, bins=35, alpha=0.45, density=True, label=evaluation.policy_name)
    plt.xlabel("Implementation shortfall (bps of arrival value)")
    plt.ylabel("Density")
    plt.title("Execution cost distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def plot_average_inventory(evaluations: list[PolicyEvaluation], output_path: str | Path) -> None:
    """Save average inventory path for each policy."""

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(9, 5))
    for evaluation in evaluations:
        inventory_paths = []
        for episode in evaluation.episodes:
            start_inventory = episode.history["inventory"][0] + episode.history["trade_size"][0]
            path = np.concatenate([[start_inventory], episode.history["inventory"]])
            inventory_paths.append(path)
        average_path = np.asarray(inventory_paths).mean(axis=0)
        plt.plot(average_path / max(average_path[0], 1e-12), label=evaluation.policy_name)
    plt.xlabel("Time bucket")
    plt.ylabel("Average remaining inventory / initial inventory")
    plt.title("Average inventory trajectories")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()

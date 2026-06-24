"""Run an Almgren-Chriss risk-aversion sanity check."""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
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

from neural_optimal_execution.config import load_project_config
from neural_optimal_execution.evaluation.metrics import PolicyEvaluation, cvar, evaluate_policy
from neural_optimal_execution.policies import AlmgrenChrissPolicy

RISK_AVERSION_GRID = (0.0, 1.25e-4, 5.0e-4, 2.0e-3, 8.0e-3)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Almgren-Chriss behavior across risk aversion values.")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to YAML config.")
    parser.add_argument("--output-dir", default="results", help="Directory for result artifacts.")
    return parser.parse_args()


def first_half_liquidation_fraction(trade_sizes: np.ndarray, parent_order: float) -> float:
    """Return the fraction of the parent order liquidated in the first half."""

    if parent_order <= 0.0 or trade_sizes.size == 0:
        return 0.0
    first_half_steps = max(1, trade_sizes.size // 2)
    return float(np.sum(trade_sizes[:first_half_steps]) / parent_order)


def average_first_half_liquidation_fraction(evaluation: PolicyEvaluation, parent_order: float) -> float:
    """Average first-half liquidation fraction across evaluated episodes."""

    fractions = [
        first_half_liquidation_fraction(episode.history["trade_size"], parent_order)
        for episode in evaluation.episodes
    ]
    return float(np.mean(fractions)) if fractions else 0.0


def average_inventory_path(evaluation: PolicyEvaluation) -> np.ndarray:
    """Return average normalized remaining inventory path including t=0."""

    paths = []
    for episode in evaluation.episodes:
        trades = episode.history["trade_size"]
        inventory = episode.history["inventory"]
        if trades.size == 0 or inventory.size == 0:
            continue
        initial_inventory = inventory[0] + trades[0]
        path = np.concatenate([[initial_inventory], inventory])
        paths.append(path / max(initial_inventory, 1e-12))
    if not paths:
        return np.zeros(0, dtype=float)
    return np.asarray(paths, dtype=float).mean(axis=0)


def summarize_ac_sanity(
    risk_aversion: float,
    evaluation: PolicyEvaluation,
    parent_order: float,
) -> dict[str, float]:
    """Build one CSV row for an AC sanity-check evaluation."""

    shortfalls = evaluation.shortfall_bps
    terminal_inventory = evaluation.terminal_inventory
    return {
        "risk_aversion": risk_aversion,
        "mean_shortfall": float(shortfalls.mean()),
        "std_shortfall": float(shortfalls.std(ddof=1)) if shortfalls.size > 1 else 0.0,
        "cvar_95": cvar(shortfalls, 0.95),
        "completion_rate": float(np.mean(np.isclose(terminal_inventory, 0.0, atol=1e-6))),
        "avg_first_half_liquidation_fraction": average_first_half_liquidation_fraction(evaluation, parent_order),
    }


def plot_inventory_by_risk_aversion(
    evaluations: list[tuple[float, PolicyEvaluation]],
    output_path: str | Path,
) -> None:
    """Save normalized average inventory paths for each risk-aversion value."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(9, 5))
    for risk_aversion, evaluation in evaluations:
        path = average_inventory_path(evaluation)
        plt.plot(np.arange(path.size), path, label=f"lambda={risk_aversion:.2g}")
    plt.xlabel("Time bucket")
    plt.ylabel("Average remaining inventory / initial inventory")
    plt.title("Almgren-Chriss inventory path by risk aversion")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def plot_cost_risk_frontier(summary: pd.DataFrame, output_path: str | Path) -> None:
    """Save the mean-cost versus risk frontier."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.plot(summary["std_shortfall"], summary["mean_shortfall"], marker="o")
    for row in summary.itertuples(index=False):
        plt.annotate(f"{row.risk_aversion:.2g}", (row.std_shortfall, row.mean_shortfall), textcoords="offset points", xytext=(5, 5))
    plt.xlabel("Std. implementation shortfall (bps)")
    plt.ylabel("Mean implementation shortfall (bps)")
    plt.title("Almgren-Chriss cost-risk frontier")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def main() -> None:
    args = parse_args()
    env_cfg, eval_cfg, _train_cfg, _ac_cfg = load_project_config(ROOT / args.config)
    output_dir = ROOT / args.output_dir

    evaluations: list[tuple[float, PolicyEvaluation]] = []
    rows: list[dict[str, float]] = []
    for risk_aversion in RISK_AVERSION_GRID:
        policy = AlmgrenChrissPolicy(risk_aversion=risk_aversion)
        print(f"Evaluating Almgren-Chriss with risk_aversion={risk_aversion:.6g}...")
        evaluation = evaluate_policy(policy, env_cfg, eval_cfg)
        evaluations.append((risk_aversion, evaluation))
        rows.append(summarize_ac_sanity(risk_aversion, evaluation, env_cfg.parent_order))

    summary = pd.DataFrame(rows)
    table_path = output_dir / "tables" / "ac_sanity_check.csv"
    table_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(table_path, index=False)

    plot_inventory_by_risk_aversion(evaluations, output_dir / "figures" / "ac_risk_aversion_inventory.png")
    plot_cost_risk_frontier(summary, output_dir / "figures" / "ac_cost_risk_frontier.png")

    print("\nAlmgren-Chriss sanity-check summary:")
    print(summary.to_string(index=False))
    print(f"\nSaved table to {table_path.relative_to(ROOT)}")
    print(f"Saved figures to {(output_dir / 'figures').relative_to(ROOT)}")


if __name__ == "__main__":
    main()

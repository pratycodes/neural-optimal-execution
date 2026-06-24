"""Run stress-test evaluations for optimal execution policies."""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import warnings
from dataclasses import replace
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
from neural_optimal_execution.environment.market_simulator import u_shaped_curve
from neural_optimal_execution.evaluation.feasibility import ensure_participation_feasible
from neural_optimal_execution.evaluation.metrics import cvar
from neural_optimal_execution.evaluation.outputs import display_path, make_run_output_dirs
from neural_optimal_execution.evaluation.stress_tests import liquidity_drought, slow_resilience, volatility_spike
from neural_optimal_execution.policies import (
    AlmgrenChrissPolicy,
    BasePolicy,
    RecalibratedAlmgrenChrissPolicy,
    TWAPPolicy,
    TrainedNeuralPolicy,
    VWAPPolicy,
    run_policy_episode,
)

SCENARIO_ORDER = (
    "base",
    "liquidity_drought",
    "volatility_spike",
    "slow_transient_decay",
    "late_day_liquidity_collapse",
    "impact_misspecification",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run execution-policy stress tests.")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to YAML config.")
    parser.add_argument("--output-dir", default="results", help="Directory for result artifacts.")
    parser.add_argument("--run-name", default=None, help="Optional run name under results/runs/<run-name>.")
    parser.add_argument("--allow-infeasible", action="store_true", help="Continue even if participation capacity is infeasible.")
    return parser.parse_args()


def resolve_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def base_volume_curve(config: ExecutionConfig) -> np.ndarray:
    """Return the baseline expected volume curve as a normalized vector."""

    if config.volume_curve is None:
        return u_shaped_curve(config.n_steps)
    curve = np.asarray(config.volume_curve, dtype=float)
    total = float(curve.sum())
    if total <= 0.0:
        raise ValueError("volume_curve must have a positive sum.")
    return curve / total


def late_day_liquidity_collapse(config: ExecutionConfig) -> ExecutionConfig:
    """Cut final-third expected volume while preserving early-bucket levels."""

    expected_volume = config.base_daily_volume * base_volume_curve(config)
    start = max(0, config.n_steps - config.n_steps // 3)
    expected_volume[start:] *= 0.25
    stressed_daily_volume = float(expected_volume.sum())
    stressed_curve = tuple((expected_volume / stressed_daily_volume).astype(float))
    return replace(config, base_daily_volume=stressed_daily_volume, volume_curve=stressed_curve)


def impact_misspecification(config: ExecutionConfig) -> ExecutionConfig:
    """Evaluate under higher impact than the baseline simulator config."""

    return replace(
        config,
        temp_impact=config.temp_impact * 2.00,
        transient_strength=config.transient_strength * 1.50,
    )


def build_scenarios(config: ExecutionConfig) -> dict[str, ExecutionConfig]:
    """Build all stress-test scenario configs."""

    return {
        "base": config,
        "liquidity_drought": liquidity_drought(config),
        "volatility_spike": volatility_spike(config),
        "slow_transient_decay": slow_resilience(config),
        "late_day_liquidity_collapse": late_day_liquidity_collapse(config),
        "impact_misspecification": impact_misspecification(config),
    }


def build_policies(ac_risk_aversion: float, model_path: Path) -> list[BasePolicy]:
    """Build fresh policy instances for one scenario evaluation."""

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


def summarize_evaluation(
    scenario: str,
    policy_name: str,
    losses: np.ndarray,
    terminal_inventory: np.ndarray,
    violations: np.ndarray,
    forced_terminal_liquidation: np.ndarray | None = None,
) -> dict[str, float | str]:
    """Build one stress-test metrics row."""

    forced = forced_terminal_liquidation if forced_terminal_liquidation is not None else np.zeros_like(violations)
    return {
        "scenario": scenario,
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


def evaluate_policy_for_scenario(
    scenario: str,
    policy: BasePolicy,
    env_config: ExecutionConfig,
    eval_config: EvaluationConfig,
) -> dict[str, float | str]:
    """Evaluate one policy under one stress scenario."""

    episodes = []
    for episode_idx in range(eval_config.n_episodes):
        seed = eval_config.seed + episode_idx
        episodes.append(run_policy_episode(policy, ExecutionEnv(env_config), seed=seed))
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
    return summarize_evaluation(
        scenario,
        policy.name,
        losses,
        terminal_inventory,
        violations,
        forced_terminal_liquidation,
    )


def plot_stress_metric(
    metrics: pd.DataFrame,
    metric: str,
    output_path: str | Path,
    *,
    ylabel: str,
    title: str,
) -> None:
    """Save a grouped scenario-by-policy bar chart."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pivot = metrics.pivot(index="scenario", columns="policy", values=metric).reindex(SCENARIO_ORDER)
    ax = pivot.plot(kind="bar", figsize=(11, 5.5), width=0.82)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Stress scenario")
    ax.set_title(title)
    ax.legend(title="Policy", fontsize=8)
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def write_interpretation(metrics: pd.DataFrame, output_path: str | Path) -> None:
    """Write a Markdown interpretation of stress-test results."""

    stress_metrics = metrics[metrics["scenario"] != "base"]
    tail_by_policy = stress_metrics.groupby("policy")["cvar_95_bps"].mean().sort_values()
    most_robust_policy = str(tail_by_policy.index[0])
    most_robust_cvar = float(tail_by_policy.iloc[0])

    neural_text = "Neural Policy was not evaluated because no saved model checkpoint was available."
    if "Neural Policy" in metrics["policy"].unique():
        neural = stress_metrics[stress_metrics["policy"] == "Neural Policy"].set_index("scenario")
        classical = stress_metrics[stress_metrics["policy"] != "Neural Policy"]
        best_classical = classical.groupby("scenario")["cvar_95_bps"].min()
        helpful = []
        weaker = []
        for scenario, row in neural.iterrows():
            if row["cvar_95_bps"] <= best_classical.loc[scenario]:
                helpful.append(scenario)
            else:
                weaker.append(scenario)
        neural_text = (
            "Neural Policy has the lowest tail risk in "
            f"{', '.join(helpful) if helpful else 'none of the stress scenarios'}."
        )
        if weaker:
            neural_text += f" Classical policies have lower CVaR in {', '.join(weaker)}."

    classical_metrics = stress_metrics[stress_metrics["policy"] != "Neural Policy"]
    classical_winners = (
        classical_metrics.sort_values(["scenario", "cvar_95_bps"])
        .groupby("scenario")
        .first()["policy"]
        .to_dict()
    )
    classical_text = ", ".join(f"{scenario}: {policy}" for scenario, policy in classical_winners.items())

    completion_failures = metrics[
        (metrics["completion_rate"] < 0.999)
        | (metrics["avg_terminal_inventory"] > 1e-6)
    ]
    participation_pressure = metrics[metrics["avg_participation_violation_shares"] > 1e-6]
    if completion_failures.empty:
        failure_text = "No policy failed to complete the parent order in these scenarios."
    else:
        failure_text = (
            "Completion failures appear in: "
            + ", ".join(f"{row.scenario}/{row.policy}" for row in completion_failures.itertuples(index=False))
            + "."
        )
    if not participation_pressure.empty:
        failure_text += (
            " Participation-limit pressure appears in: "
            + ", ".join(f"{row.scenario}/{row.policy}" for row in participation_pressure.itertuples(index=False))
            + "."
        )

    text = f"""# Stress-Test Interpretation

Stress tests use the existing simulator with config-level shocks. They are synthetic robustness diagnostics, not live-market performance claims.

## Tail-Risk Robustness

Across non-base stress scenarios, `{most_robust_policy}` has the lowest average CVaR 95 at `{most_robust_cvar:.2f}` bps.

## Neural Policy

{neural_text}

## Classical Baselines

Among classical policies, the lowest-CVaR policy by stress scenario is: {classical_text}.

## Liquidity Stress And Completion

{failure_text}
"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    env_config, eval_config, _train_config, ac_config = load_project_config(resolve_path(args.config))
    scenario_configs = build_scenarios(env_config)
    all_feasible = True
    for scenario_name, scenario_config in scenario_configs.items():
        all_feasible = (
            ensure_participation_feasible(
                scenario_config,
                label=f"{args.config}:{scenario_name}",
                allow_infeasible=args.allow_infeasible,
            )
            and all_feasible
        )
    if not all_feasible:
        raise SystemExit(1)
    output_dirs = make_run_output_dirs(resolve_path(args.output_dir), args.run_name)
    output_dir = output_dirs.root
    model_path = output_dirs.models / "neural_policy.pt"
    if args.run_name:
        print(f"Using run output directory: {display_path(output_dir, ROOT)}")

    if not model_path.exists():
        print(f"Skipping Neural Policy because no saved model exists at {display_path(model_path, ROOT)}.")

    rows: list[dict[str, float | str]] = []
    for scenario_name, scenario_config in scenario_configs.items():
        for policy in build_policies(ac_config.risk_aversion, model_path):
            print(f"Scenario {scenario_name}: evaluating {policy.name}...")
            rows.append(evaluate_policy_for_scenario(scenario_name, policy, scenario_config, eval_config))

    metrics = pd.DataFrame(rows)
    table_path = output_dirs.tables / "stress_test_metrics.csv"
    metrics.to_csv(table_path, index=False)

    plot_stress_metric(
        metrics,
        "cvar_95_bps",
        output_dirs.figures / "stress_test_cvar.png",
        ylabel="CVaR 95 implementation shortfall (bps)",
        title="Stress-test CVaR by policy",
    )
    plot_stress_metric(
        metrics,
        "mean_shortfall_bps",
        output_dirs.figures / "stress_test_mean_shortfall.png",
        ylabel="Mean implementation shortfall (bps)",
        title="Stress-test mean shortfall by policy",
    )
    plot_stress_metric(
        metrics,
        "worst_shortfall_bps",
        output_dirs.figures / "stress_test_worst_shortfall.png",
        ylabel="Worst implementation shortfall (bps)",
        title="Stress-test worst shortfall by policy",
    )
    interpretation_path = output_dir / "stress_test_interpretation.md"
    write_interpretation(metrics, interpretation_path)

    print("\nStress-test summary:")
    print(metrics.to_string(index=False))
    print(f"\nSaved table to {display_path(table_path, ROOT)}")
    print(f"Saved figures to {display_path(output_dirs.figures, ROOT)}")
    print(f"Saved interpretation to {display_path(interpretation_path, ROOT)}")


if __name__ == "__main__":
    main()

"""Generate neural policy heatmap diagnostics."""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
MPL_CACHE_DIR = Path(tempfile.gettempdir()) / "neural_optimal_execution_matplotlib"
XDG_CACHE_DIR = Path(tempfile.gettempdir()) / "neural_optimal_execution_cache"
MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
XDG_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR))
os.environ.setdefault("XDG_CACHE_HOME", str(XDG_CACHE_DIR))

import matplotlib.pyplot as plt

from neural_optimal_execution.config import ExecutionConfig, load_project_config
from neural_optimal_execution.policies.neural_policy import MLPExecutionPolicy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot neural policy trade-fraction heatmaps.")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to YAML config.")
    parser.add_argument("--model-path", default="results/models/neural_policy.pt", help="Path to saved neural policy.")
    parser.add_argument("--output-dir", default="results", help="Directory for result artifacts.")
    return parser.parse_args()


def resolve_path(path: str | Path) -> Path:
    """Resolve a CLI path relative to the project root."""

    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def missing_model_message(model_path: str | Path, config_path: str | Path) -> str:
    """Return a clear user-facing message for a missing neural checkpoint."""

    return (
        f"Saved neural policy not found at {model_path}.\n"
        "Train the neural policy first with:\n"
        f"python experiments/train_neural_policy.py --config {config_path}"
    )


def display_path(path: Path) -> Path:
    """Return a project-relative path when possible."""

    try:
        return path.relative_to(ROOT)
    except ValueError:
        return path


def baseline_states(n: int, env_config: ExecutionConfig) -> np.ndarray:
    """Return economically neutral normalized states in the training convention."""

    states = np.zeros((n, 9), dtype=np.float32)
    states[:, 0] = 0.50  # inventory fraction
    states[:, 1] = 0.50  # time remaining
    states[:, 2] = 0.50  # time elapsed
    states[:, 3] = 0.00  # price equals initial price
    states[:, 4] = env_config.base_volatility / max(env_config.base_volatility, 1e-12)
    states[:, 5] = 1.00  # current volume equals normal average bucket volume
    states[:, 6] = 1.00  # expected volume equals normal average bucket volume
    states[:, 7] = env_config.temp_impact / max(env_config.temp_impact, 1e-12)
    states[:, 8] = 0.00  # no transient impact
    return states


def predict_trade_fraction(model: MLPExecutionPolicy, states: np.ndarray) -> np.ndarray:
    """Predict the fraction of max allowable trade for normalized states."""

    with torch.no_grad():
        tensor = torch.tensor(states, dtype=torch.float32)
        return model(tensor).cpu().numpy()


def predict_grid(
    model: MLPExecutionPolicy,
    env_config: ExecutionConfig,
    x_values: np.ndarray,
    y_values: np.ndarray,
    x_index: int,
    y_index: int,
    *,
    couple_time_elapsed: bool = False,
) -> np.ndarray:
    """Predict a heatmap grid while varying two state dimensions."""

    xx, yy = np.meshgrid(x_values, y_values)
    states = baseline_states(xx.size, env_config)
    states[:, x_index] = xx.ravel()
    states[:, y_index] = yy.ravel()
    if couple_time_elapsed:
        states[:, 2] = 1.0 - states[:, 1]
    return predict_trade_fraction(model, states).reshape(len(y_values), len(x_values))


def plot_heatmap(
    values: np.ndarray,
    x_values: np.ndarray,
    y_values: np.ndarray,
    output_path: str | Path,
    *,
    xlabel: str,
    ylabel: str,
    title: str,
) -> None:
    """Save one trade-fraction heatmap."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5.5))
    image = plt.imshow(
        values,
        origin="lower",
        aspect="auto",
        extent=[float(x_values[0]), float(x_values[-1]), float(y_values[0]), float(y_values[-1])],
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
    )
    plt.colorbar(image, label="Predicted fraction of max allowable trade")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def scalar_prediction(
    model: MLPExecutionPolicy,
    env_config: ExecutionConfig,
    *,
    inventory: float = 0.50,
    time_remaining: float = 0.50,
    volatility: float = 1.00,
    liquidity: float = 1.00,
    transient: float = 0.00,
) -> float:
    """Predict one trade fraction for interpretation comparisons."""

    state = baseline_states(1, env_config)
    state[0, 0] = inventory
    state[0, 1] = time_remaining
    state[0, 2] = 1.0 - time_remaining
    state[0, 4] = volatility
    state[0, 5] = liquidity
    state[0, 8] = transient
    return float(predict_trade_fraction(model, state)[0])


def directional_sentence(
    higher_value: float,
    lower_value: float,
    positive_text: str,
    negative_text: str,
    flat_text: str,
    tolerance: float = 0.01,
) -> str:
    """Return a short interpretation sentence from two predictions."""

    delta = higher_value - lower_value
    if delta > tolerance:
        return positive_text
    if delta < -tolerance:
        return negative_text
    return flat_text


def write_interpretation(model: MLPExecutionPolicy, env_config: ExecutionConfig, output_path: str | Path) -> None:
    """Write a Markdown interpretation of policy heatmap diagnostics."""

    high_inventory_short_time = scalar_prediction(model, env_config, inventory=0.90, time_remaining=0.15)
    low_inventory_short_time = scalar_prediction(model, env_config, inventory=0.20, time_remaining=0.15)
    high_inventory_long_time = scalar_prediction(model, env_config, inventory=0.90, time_remaining=0.85)
    high_liquidity = scalar_prediction(model, env_config, liquidity=2.00)
    low_liquidity = scalar_prediction(model, env_config, liquidity=0.40)
    high_transient = scalar_prediction(model, env_config, transient=5.0e-4, time_remaining=0.50)
    low_transient = scalar_prediction(model, env_config, transient=0.00, time_remaining=0.50)
    high_volatility = scalar_prediction(model, env_config, volatility=2.50)
    low_volatility = scalar_prediction(model, env_config, volatility=0.50)

    inventory_sentence = directional_sentence(
        high_inventory_short_time,
        low_inventory_short_time,
        "At short time remaining, the policy increases the trade fraction when inventory is high.",
        "At short time remaining, the policy unexpectedly reduces the trade fraction when inventory is high.",
        "At short time remaining, the policy is nearly flat across low and high inventory.",
    )
    urgency_sentence = directional_sentence(
        high_inventory_short_time,
        high_inventory_long_time,
        "For high inventory, the policy accelerates as time remaining gets short.",
        "For high inventory, the policy unexpectedly slows down as time remaining gets short.",
        "For high inventory, the policy is nearly flat across time remaining.",
    )
    liquidity_sentence = directional_sentence(
        high_liquidity,
        low_liquidity,
        "The policy trades a larger fraction of the max allowable trade when current liquidity is good.",
        "The policy trades a smaller fraction of the max allowable trade when current liquidity is good.",
        "The policy fraction is nearly flat across the tested liquidity range; in share terms, higher liquidity still increases the max allowable trade.",
    )
    transient_sentence = directional_sentence(
        high_transient,
        low_transient,
        "The policy increases the trade fraction when transient impact is high, which is not the expected economic response.",
        "The policy slows down when transient impact is high, matching the expected resilience-aware behavior.",
        "The policy is nearly flat across the tested transient-impact range.",
    )
    volatility_sentence = directional_sentence(
        high_volatility,
        low_volatility,
        "The policy increases the trade fraction under high volatility, consistent with reducing price-risk exposure.",
        "The policy slows down under high volatility, which may indicate it is prioritizing impact control over price-risk reduction.",
        "The policy is nearly flat across the tested volatility range.",
    )

    text = f"""# Neural Policy Heatmap Interpretation

The heatmaps show the saved neural policy's raw sigmoid output: the predicted fraction of the current maximum allowable trade. Actual shares are this fraction multiplied by `min(remaining_inventory, participation_rate * current_volume)`.

## Inventory And Time

- High inventory, short time remaining: `{high_inventory_short_time:.3f}`
- Low inventory, short time remaining: `{low_inventory_short_time:.3f}`
- High inventory, long time remaining: `{high_inventory_long_time:.3f}`

{inventory_sentence} {urgency_sentence}

## Liquidity And Volatility

- High liquidity, neutral volatility: `{high_liquidity:.3f}`
- Low liquidity, neutral volatility: `{low_liquidity:.3f}`
- High volatility, neutral liquidity: `{high_volatility:.3f}`
- Low volatility, neutral liquidity: `{low_volatility:.3f}`

{liquidity_sentence} {volatility_sentence}

## Transient Impact

- High transient impact: `{high_transient:.3f}`
- Zero transient impact: `{low_transient:.3f}`

{transient_sentence}

## Caveat

These diagnostics vary two normalized state dimensions at a time and hold the rest fixed at neutral values. They are local slices of the policy, not a full causal explanation of execution performance.
"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")


def main() -> int:
    args = parse_args()
    config_path = resolve_path(args.config)
    model_path = resolve_path(args.model_path)
    output_dir = resolve_path(args.output_dir)
    env_config, _eval_config, _train_config, _ac_config = load_project_config(config_path)
    if not model_path.exists():
        print(missing_model_message(display_path(model_path), args.config), file=sys.stderr)
        return 1

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning, message="You are using `torch.load`.*")
        model = MLPExecutionPolicy.load(model_path)
    model.eval()

    inventory_values = np.linspace(0.05, 1.00, 60, dtype=np.float32)
    time_values = np.linspace(0.05, 1.00, 60, dtype=np.float32)
    inventory_time = predict_grid(
        model,
        env_config,
        time_values,
        inventory_values,
        x_index=1,
        y_index=0,
        couple_time_elapsed=True,
    )
    plot_heatmap(
        inventory_time,
        time_values,
        inventory_values,
        output_dir / "figures" / "policy_heatmap_inventory_time.png",
        xlabel="Time remaining fraction",
        ylabel="Inventory fraction",
        title="Neural policy: inventory vs time remaining",
    )

    liquidity_values = np.linspace(0.25, 2.50, 60, dtype=np.float32)
    volatility_values = np.linspace(0.50, 3.00, 60, dtype=np.float32)
    liquidity_volatility = predict_grid(
        model,
        env_config,
        liquidity_values,
        volatility_values,
        x_index=5,
        y_index=4,
    )
    plot_heatmap(
        liquidity_volatility,
        liquidity_values,
        volatility_values,
        output_dir / "figures" / "policy_heatmap_liquidity_volatility.png",
        xlabel="Current volume / average bucket volume",
        ylabel="Volatility / base volatility",
        title="Neural policy: liquidity vs volatility",
    )

    transient_values = np.linspace(0.00, 5.0e-4, 60, dtype=np.float32)
    transient_time = predict_grid(
        model,
        env_config,
        time_values,
        transient_values,
        x_index=1,
        y_index=8,
        couple_time_elapsed=True,
    )
    plot_heatmap(
        transient_time,
        time_values,
        transient_values,
        output_dir / "figures" / "policy_heatmap_transient_impact.png",
        xlabel="Time remaining fraction",
        ylabel="Transient impact / initial price",
        title="Neural policy: transient impact vs time remaining",
    )

    interpretation_path = output_dir / "policy_heatmap_interpretation.md"
    write_interpretation(model, env_config, interpretation_path)

    print("Saved policy heatmaps to:")
    print(f"- {(output_dir / 'figures' / 'policy_heatmap_inventory_time.png').relative_to(ROOT)}")
    print(f"- {(output_dir / 'figures' / 'policy_heatmap_liquidity_volatility.png').relative_to(ROOT)}")
    print(f"- {(output_dir / 'figures' / 'policy_heatmap_transient_impact.png').relative_to(ROOT)}")
    print(f"Saved interpretation to {interpretation_path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

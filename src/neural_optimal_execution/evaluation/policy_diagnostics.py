"""Diagnostics for learned execution-policy state dependence."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from neural_optimal_execution.policies.neural_policy import MLPExecutionPolicy

STATE_FEATURES = (
    "inventory_fraction",
    "time_remaining",
    "time_elapsed",
    "price_return",
    "volatility",
    "current_volume",
    "expected_volume",
    "temporary_impact",
    "transient_impact",
)


def state_ablation_sensitivity(
    model: MLPExecutionPolicy,
    states: np.ndarray,
    reference_state: np.ndarray | None = None,
) -> pd.DataFrame:
    """Measure output changes when each state feature is replaced by a reference value."""

    state_array = np.asarray(states, dtype=np.float32)
    if state_array.ndim != 2 or state_array.shape[1] != len(STATE_FEATURES):
        raise ValueError(f"states must have shape (n, {len(STATE_FEATURES)}).")
    if state_array.shape[0] == 0:
        raise ValueError("states cannot be empty.")
    reference = (
        np.median(state_array, axis=0)
        if reference_state is None
        else np.asarray(reference_state, dtype=np.float32)
    )
    if reference.shape != (len(STATE_FEATURES),):
        raise ValueError(f"reference_state must have shape ({len(STATE_FEATURES)},).")

    baseline = _predict(model, state_array)
    rows: list[dict[str, float | str]] = []
    for feature_index, feature_name in enumerate(STATE_FEATURES):
        ablated = state_array.copy()
        ablated[:, feature_index] = reference[feature_index]
        delta = _predict(model, ablated) - baseline
        rows.append(
            {
                "feature": feature_name,
                "mean_absolute_fraction_change": float(np.mean(np.abs(delta))),
                "max_absolute_fraction_change": float(np.max(np.abs(delta))),
                "signed_mean_fraction_change": float(np.mean(delta)),
                "baseline_fraction_std": float(np.std(baseline)),
            }
        )
    return pd.DataFrame(rows).sort_values(
        "mean_absolute_fraction_change",
        ascending=False,
    ).reset_index(drop=True)


def _predict(model: MLPExecutionPolicy, states: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        tensor = torch.tensor(states, dtype=torch.float32)
        return model(tensor).cpu().numpy()

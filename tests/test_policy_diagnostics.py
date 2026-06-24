import numpy as np
import torch

from neural_optimal_execution.evaluation.policy_diagnostics import state_ablation_sensitivity
from neural_optimal_execution.policies.neural_policy import MLPExecutionPolicy


def test_state_ablation_identifies_the_feature_used_by_policy():
    model = MLPExecutionPolicy(input_dim=9, hidden_sizes=())
    with torch.no_grad():
        model.network[0].weight.zero_()
        model.network[0].weight[0, 0] = 4.0
        model.network[0].bias.zero_()
    states = np.zeros((5, 9), dtype=np.float32)
    states[:, 0] = np.linspace(-1.0, 1.0, 5)

    diagnostics = state_ablation_sensitivity(model, states, reference_state=np.zeros(9))

    inventory_row = diagnostics[diagnostics["feature"] == "inventory_fraction"].iloc[0]
    other_rows = diagnostics[diagnostics["feature"] != "inventory_fraction"]
    assert inventory_row["mean_absolute_fraction_change"] > 0.0
    assert (other_rows["mean_absolute_fraction_change"] == 0.0).all()

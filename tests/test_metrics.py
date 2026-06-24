import numpy as np
import torch

from neural_optimal_execution.evaluation.metrics import completion_rate, constraint_summary_metrics, cvar
from neural_optimal_execution.training.losses import empirical_cvar


def test_cvar_right_tail():
    values = np.array([1.0, 2.0, 3.0, 100.0])
    assert cvar(values, 0.75) == 100.0


def test_cvar_uses_average_of_tail_losses_on_toy_vector():
    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert cvar(values, 0.60) == 4.5


def test_training_empirical_cvar_matches_toy_tail_average():
    losses = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    assert empirical_cvar(losses, 0.60).item() == 4.5


def test_constraint_summary_metrics_include_parent_order_fractions():
    metrics = constraint_summary_metrics(
        terminal_inventory=np.array([0.0, 10.0]),
        participation_violations=np.array([0.0, 4.0]),
        forced_terminal_liquidation=np.array([0.0, 8.0]),
        parent_order=20.0,
    )

    assert metrics["avg_terminal_inventory"] == 5.0
    assert metrics["avg_participation_violation_shares"] == 2.0
    assert metrics["forced_terminal_liquidation_shares"] == 4.0
    assert metrics["terminal_inventory_fraction"] == 0.25
    assert metrics["participation_violation_fraction_of_parent_order"] == 0.1
    assert metrics["forced_terminal_liquidation_fraction"] == 0.2


def test_completion_rate_uses_parent_order_relative_tolerance():
    terminal_inventory = np.array([0.0, 0.75, 2.0])

    rate = completion_rate(terminal_inventory, parent_order=1_000_000.0, tolerance_fraction=1.0e-6)

    assert rate == 2.0 / 3.0

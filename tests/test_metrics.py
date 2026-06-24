import numpy as np
import torch

from neural_optimal_execution.evaluation.metrics import cvar
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

"""Loss functions for neural optimal execution."""

from __future__ import annotations

import torch


def empirical_cvar(losses: torch.Tensor, level: float = 0.95) -> torch.Tensor:
    """Differentiable empirical CVaR approximation for the right tail."""

    if losses.ndim != 1:
        losses = losses.reshape(-1)
    threshold = torch.quantile(losses.detach(), level)
    tail_losses = losses[losses >= threshold]
    if tail_losses.numel() == 0:
        return threshold
    return tail_losses.mean()


def mean_variance_cvar_objective(
    losses: torch.Tensor,
    terminal_inventory_fraction: torch.Tensor,
    lambda_var: float = 0.5,
    alpha_cvar: float = 0.0,
    beta_terminal: float = 10.0,
    cvar_level: float = 0.95,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Combine mean cost, variance, CVaR, and terminal inventory penalties."""

    mean_loss = losses.mean()
    variance = losses.var(unbiased=False)
    cvar = empirical_cvar(losses, cvar_level)
    terminal_penalty = (terminal_inventory_fraction**2).mean()
    objective = mean_loss + lambda_var * variance + alpha_cvar * cvar + beta_terminal * terminal_penalty
    stats = {
        "objective": float(objective.detach().cpu()),
        "mean_loss": float(mean_loss.detach().cpu()),
        "variance": float(variance.detach().cpu()),
        "cvar": float(cvar.detach().cpu()),
        "terminal_penalty": float(terminal_penalty.detach().cpu()),
    }
    return objective, stats

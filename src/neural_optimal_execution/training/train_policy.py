"""Direct neural policy optimization through a differentiable simulator."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import optim

from neural_optimal_execution.config import ExecutionConfig, TrainingConfig
from neural_optimal_execution.environment.market_simulator import MarketSimulator
from neural_optimal_execution.policies.neural_policy import MLPExecutionPolicy
from neural_optimal_execution.training.losses import mean_variance_cvar_objective


def _make_torch_market_paths(
    env_config: ExecutionConfig,
    batch_size: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Sample vectorized synthetic market paths in torch."""

    n_steps = env_config.n_steps
    simulator = MarketSimulator(env_config)
    expected_volume_np = env_config.base_daily_volume * simulator.expected_volume_curve
    expected_vol_np = simulator.expected_volatility_curve
    expected_volume = torch.tensor(expected_volume_np, dtype=torch.float32, device=device).expand(batch_size, n_steps)
    expected_vol = torch.tensor(
        expected_vol_np / max(float(expected_vol_np.mean()), 1e-12),
        dtype=torch.float32,
        device=device,
    ).expand(batch_size, n_steps)

    # Independent regime sampling is sufficient for the training MVP. The numpy
    # evaluation environment uses a Markov regime chain.
    regime_probs = torch.tensor([0.75, 0.15, 0.10], dtype=torch.float32, device=device)
    regimes = torch.multinomial(regime_probs, num_samples=batch_size * n_steps, replacement=True).reshape(batch_size, n_steps)
    volume_mult = torch.tensor([1.00, 1.60, 0.45], dtype=torch.float32, device=device)[regimes]
    volatility_mult = torch.tensor([1.00, 1.25, 1.90], dtype=torch.float32, device=device)[regimes]
    impact_mult = torch.tensor([1.00, 0.70, 2.25], dtype=torch.float32, device=device)[regimes]

    volume_noise = env_config.volume_noise
    volume = expected_volume * volume_mult
    volume = volume * torch.exp(volume_noise * torch.randn(batch_size, n_steps, device=device) - 0.5 * volume_noise**2)
    volume = torch.clamp(volume, min=1.0)

    volatility_noise = env_config.volatility_noise
    volatility = env_config.base_volatility * expected_vol * volatility_mult
    volatility = volatility * torch.exp(volatility_noise * torch.randn(batch_size, n_steps, device=device) - 0.5 * volatility_noise**2)

    temp_impact = env_config.temp_impact * impact_mult
    temp_impact = temp_impact * torch.exp(0.15 * torch.randn(batch_size, n_steps, device=device) - 0.5 * 0.15**2)

    dt = env_config.horizon / env_config.n_steps
    returns = volatility * (dt**0.5) * torch.randn(batch_size, n_steps, device=device)
    transient_noise = env_config.transient_noise * torch.randn(batch_size, n_steps, device=device)
    return {
        "expected_volume": expected_volume,
        "volume": volume,
        "volatility": volatility,
        "temp_impact": temp_impact,
        "returns": returns,
        "transient_noise": transient_noise,
    }


def simulate_batch(
    policy: MLPExecutionPolicy,
    env_config: ExecutionConfig,
    train_config: TrainingConfig,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    """Simulate a batch of episodes and return normalized shortfall losses."""

    batch_size = train_config.batch_size
    n_steps = env_config.n_steps
    market = _make_torch_market_paths(env_config, batch_size, device)

    x0 = torch.full((batch_size,), float(env_config.parent_order), dtype=torch.float32, device=device)
    initial_price = float(env_config.initial_price)
    benchmark = x0 * initial_price
    inventory = x0.clone()
    price = torch.full((batch_size,), initial_price, dtype=torch.float32, device=device)
    cash = torch.zeros(batch_size, dtype=torch.float32, device=device)
    transient = torch.zeros(batch_size, dtype=torch.float32, device=device)
    average_bucket_volume = env_config.base_daily_volume / env_config.n_steps

    participation_used = []
    for t in range(n_steps):
        volume_t = market["volume"][:, t]
        expected_volume_t = market["expected_volume"][:, t]
        volatility_t = market["volatility"][:, t]
        temp_impact_t = market["temp_impact"][:, t]
        max_trade = torch.minimum(inventory, env_config.participation_rate * volume_t)
        state = torch.stack(
            [
                inventory / x0,
                torch.full_like(inventory, (n_steps - t) / n_steps),
                torch.full_like(inventory, t / n_steps),
                price / initial_price - 1.0,
                volatility_t / max(env_config.base_volatility, 1e-12),
                volume_t / max(average_bucket_volume, 1e-12),
                expected_volume_t / max(average_bucket_volume, 1e-12),
                temp_impact_t / max(env_config.temp_impact, 1e-12),
                transient / max(initial_price, 1e-12),
            ],
            dim=1,
        )
        fraction = policy(state)
        desired_trade = max_trade * fraction
        if env_config.terminal_liquidation and t == n_steps - 1:
            trade_size = inventory
        else:
            trade_size = desired_trade
        execution_price = price - temp_impact_t * trade_size - transient
        cash = cash + trade_size * execution_price
        inventory = inventory - trade_size
        transient = env_config.transient_decay * transient + env_config.transient_strength * trade_size + market["transient_noise"][:, t]
        price = torch.clamp(price * (1.0 + market["returns"][:, t]) - env_config.permanent_impact * trade_size, min=1e-8)
        participation_used.append((trade_size / torch.clamp(volume_t, min=1.0)).mean())

    losses = (benchmark - cash) / benchmark
    terminal_inventory_fraction = inventory / x0
    diagnostics = {
        "avg_terminal_inventory_fraction": float(terminal_inventory_fraction.mean().detach().cpu()),
        "avg_participation_used": float(torch.stack(participation_used).mean().detach().cpu()),
    }
    return losses, terminal_inventory_fraction, diagnostics


def train_neural_policy(
    env_config: ExecutionConfig,
    train_config: TrainingConfig,
    output_model_path: str | Path | None = None,
    output_log_path: str | Path | None = None,
) -> tuple[MLPExecutionPolicy, pd.DataFrame]:
    """Train an MLP policy through Monte Carlo simulation."""

    torch.manual_seed(train_config.seed)
    np.random.seed(train_config.seed)
    device = torch.device(train_config.device)
    if device.type == "cpu":
        # Small Monte Carlo batches are faster and more reproducible with one thread.
        torch.set_num_threads(1)
    policy = MLPExecutionPolicy(input_dim=9, hidden_sizes=train_config.hidden_sizes).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=train_config.learning_rate)
    rows: list[dict[str, float | int]] = []

    for epoch in range(1, train_config.epochs + 1):
        optimizer.zero_grad(set_to_none=True)
        losses, terminal_inventory, diagnostics = simulate_batch(policy, env_config, train_config, device)
        objective, stats = mean_variance_cvar_objective(
            losses=losses,
            terminal_inventory_fraction=terminal_inventory,
            lambda_var=train_config.lambda_var,
            alpha_cvar=train_config.alpha_cvar,
            beta_terminal=train_config.beta_terminal,
            cvar_level=train_config.cvar_level,
        )
        objective.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), train_config.grad_clip_norm)
        optimizer.step()
        rows.append({"epoch": epoch, **stats, **diagnostics})

    log_df = pd.DataFrame(rows)
    if output_model_path is not None:
        output_model_path = Path(output_model_path)
        output_model_path.parent.mkdir(parents=True, exist_ok=True)
        policy.save(output_model_path)
    if output_log_path is not None:
        output_log_path = Path(output_log_path)
        output_log_path.parent.mkdir(parents=True, exist_ok=True)
        log_df.to_csv(output_log_path, index=False)
    return policy.cpu(), log_df

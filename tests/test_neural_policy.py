import torch

from neural_optimal_execution.config import ExecutionConfig, TrainingConfig
from neural_optimal_execution.environment import ExecutionEnv
from neural_optimal_execution.policies.neural_policy import MLPExecutionPolicy, TrainedNeuralPolicy
from neural_optimal_execution.training.train_policy import simulate_batch


def test_trained_neural_policy_adapter_returns_valid_action():
    env = ExecutionEnv(ExecutionConfig(parent_order=1000.0, n_steps=5, base_daily_volume=100000.0))
    state = env.reset(seed=123)
    policy = TrainedNeuralPolicy(MLPExecutionPolicy(input_dim=env.state_dim, hidden_sizes=(8,)))
    action = policy.act(state, env)
    assert 0.0 <= action <= env.max_trade_size()


def test_neural_policy_action_is_bounded_by_inventory_and_participation():
    env = ExecutionEnv(ExecutionConfig(parent_order=1000.0, n_steps=5, base_daily_volume=100000.0))
    state = env.reset(seed=123)
    policy = TrainedNeuralPolicy(MLPExecutionPolicy(input_dim=env.state_dim, hidden_sizes=(8,)))

    action = policy.act(state, env)

    assert action >= 0.0
    assert action <= env.inventory
    assert action <= env.config.participation_rate * env.market_path.volume[env.step_index]


def test_differentiable_simulator_forces_terminal_completion_when_configured():
    env_cfg = ExecutionConfig(parent_order=1000.0, n_steps=4, base_daily_volume=100000.0, terminal_liquidation=True)
    train_cfg = TrainingConfig(batch_size=3, epochs=1, hidden_sizes=(8,), seed=123)
    torch.manual_seed(train_cfg.seed)
    policy = MLPExecutionPolicy(input_dim=9, hidden_sizes=train_cfg.hidden_sizes)

    _losses, terminal_inventory_fraction, _diagnostics = simulate_batch(
        policy,
        env_cfg,
        train_cfg,
        torch.device("cpu"),
    )

    assert torch.allclose(terminal_inventory_fraction, torch.zeros_like(terminal_inventory_fraction))

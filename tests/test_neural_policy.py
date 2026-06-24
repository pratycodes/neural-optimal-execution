import torch

from neural_optimal_execution.config import ExecutionConfig, TrainingConfig
from neural_optimal_execution.environment import ExecutionEnv
from neural_optimal_execution.environment.market_simulator import MarketPath
from neural_optimal_execution.policies.neural_policy import MLPExecutionPolicy, TrainedNeuralPolicy
from neural_optimal_execution.training.train_policy import _sample_markov_regimes, simulate_batch


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


def test_training_regimes_use_markov_chain_with_normal_initial_state():
    torch.manual_seed(123)
    regimes = _sample_markov_regimes(batch_size=4000, n_steps=3, device=torch.device("cpu"))

    assert torch.equal(regimes[:, 0], torch.zeros(4000, dtype=torch.long))
    normal_persistence = (regimes[:, 1] == 0).float().mean().item()
    assert 0.87 <= normal_persistence <= 0.93


def test_strict_neural_action_respects_expected_completion_floor():
    env = ExecutionEnv(
        ExecutionConfig(
            parent_order=100.0,
            n_steps=2,
            base_daily_volume=1000.0,
            participation_rate=0.10,
            terminal_liquidation=False,
        )
    )
    env.market_path = MarketPath(
        expected_volume=torch.tensor([500.0, 500.0]).numpy(),
        volume=torch.tensor([500.0, 500.0]).numpy(),
        volatility=torch.zeros(2).numpy(),
        temp_impact=torch.zeros(2).numpy(),
        returns=torch.zeros(2).numpy(),
        transient_noise=torch.zeros(2).numpy(),
        regimes=torch.zeros(2, dtype=torch.long).numpy(),
    )
    zero_fraction_model = MLPExecutionPolicy(input_dim=env.state_dim, hidden_sizes=())
    with torch.no_grad():
        zero_fraction_model.network[0].weight.zero_()
        zero_fraction_model.network[0].bias.fill_(-100.0)
    policy = TrainedNeuralPolicy(zero_fraction_model)

    action = policy.act(env.observe(), env)

    assert action == 50.0

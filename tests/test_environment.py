import numpy as np

from neural_optimal_execution.config import ExecutionConfig
from neural_optimal_execution.environment import ExecutionEnv
from neural_optimal_execution.environment.market_simulator import MarketPath
from neural_optimal_execution.policies import BasePolicy, TWAPPolicy, run_policy_episode


class NeverTradePolicy(BasePolicy):
    name = "Never Trade"

    def act(self, state, env):
        return 0.0


class OverTradePolicy(BasePolicy):
    name = "Over Trade"

    def act(self, state, env):
        return env.inventory * 10.0


def run_twap_on_deterministic_path(returns, temp_impact):
    cfg = ExecutionConfig(
        parent_order=1000.0,
        n_steps=4,
        initial_price=100.0,
        base_volatility=0.0,
        volatility_noise=0.0,
        base_daily_volume=1_000_000.0,
        volume_noise=0.0,
        participation_rate=1.0,
        temp_impact=temp_impact,
        permanent_impact=0.0,
        transient_strength=0.0,
        transient_noise=0.0,
        terminal_liquidation=True,
    )
    env = ExecutionEnv(cfg)
    env.reset(seed=123)
    returns = np.asarray(returns, dtype=np.float64)
    env.market_path = MarketPath(
        expected_volume=np.full(cfg.n_steps, cfg.base_daily_volume / cfg.n_steps),
        volume=np.full(cfg.n_steps, cfg.base_daily_volume / cfg.n_steps),
        volatility=np.zeros(cfg.n_steps),
        temp_impact=np.full(cfg.n_steps, temp_impact),
        returns=returns,
        transient_noise=np.zeros(cfg.n_steps),
        regimes=np.zeros(cfg.n_steps, dtype=np.int64),
    )

    policy = TWAPPolicy()
    state = env.observe()
    policy.reset(env)
    done = False
    while not done:
        state, done, _info = env.step(policy.act(state, env))
    return env.result()


def test_twap_episode_completes_with_terminal_liquidation():
    cfg = ExecutionConfig(parent_order=1000.0, n_steps=5, base_daily_volume=100000.0)
    env = ExecutionEnv(cfg)
    result = run_policy_episode(TWAPPolicy(), env, seed=123)
    assert result.terminal_inventory == 0.0
    assert len(result.history["trade_size"]) == cfg.n_steps
    assert result.normalized_shortfall_bps == result.normalized_shortfall_bps


def test_observation_dimension():
    cfg = ExecutionConfig(parent_order=1000.0, n_steps=5)
    env = ExecutionEnv(cfg)
    state = env.reset(seed=123)
    assert state.shape == (env.state_dim,)


def test_environment_never_allows_negative_inventory():
    cfg = ExecutionConfig(parent_order=1000.0, n_steps=5, base_daily_volume=100000.0)
    env = ExecutionEnv(cfg)
    result = run_policy_episode(OverTradePolicy(), env, seed=123)

    assert result.terminal_inventory >= 0.0
    assert (result.history["inventory"] >= 0.0).all()


def test_terminal_liquidation_completes_even_if_policy_does_not_trade():
    cfg = ExecutionConfig(parent_order=1000.0, n_steps=5, base_daily_volume=100000.0, terminal_liquidation=True)
    env = ExecutionEnv(cfg)
    result = run_policy_episode(NeverTradePolicy(), env, seed=123)

    assert result.terminal_inventory == 0.0
    assert result.history["trade_size"][-1] == cfg.parent_order
    assert result.history["forced_terminal_liquidation"][-1] == cfg.parent_order
    assert result.history["forced_terminal_liquidation"][:-1].sum() == 0.0


def test_zero_drift_zero_impact_has_zero_implementation_shortfall():
    result = run_twap_on_deterministic_path(returns=[0.0, 0.0, 0.0, 0.0], temp_impact=0.0)

    assert np.isclose(result.shortfall, 0.0)
    assert np.isclose(result.normalized_shortfall_bps, 0.0)


def test_positive_temporary_impact_has_positive_sell_shortfall_and_bps_scaling():
    result = run_twap_on_deterministic_path(returns=[0.0, 0.0, 0.0, 0.0], temp_impact=1.0e-3)

    benchmark_value = 1000.0 * 100.0
    expected_shortfall = 4 * (250.0 * 1.0e-3 * 250.0)
    assert np.isclose(result.shortfall, expected_shortfall)
    assert np.isclose(result.normalized_shortfall_bps, 10_000.0 * result.shortfall / benchmark_value)
    assert result.shortfall > 0.0


def test_upward_price_drift_can_create_negative_sell_shortfall_without_impact():
    result = run_twap_on_deterministic_path(returns=[0.01, 0.01, 0.01, 0.01], temp_impact=0.0)

    assert result.shortfall < 0.0
    assert result.normalized_shortfall_bps < 0.0


def test_downward_price_drift_creates_positive_sell_shortfall_without_impact():
    result = run_twap_on_deterministic_path(returns=[-0.01, -0.01, -0.01, -0.01], temp_impact=0.0)

    assert result.shortfall > 0.0
    assert result.normalized_shortfall_bps > 0.0

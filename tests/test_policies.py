import numpy as np

from neural_optimal_execution.config import ExecutionConfig
from neural_optimal_execution.environment import ExecutionEnv
from neural_optimal_execution.environment.market_simulator import MarketPath
from neural_optimal_execution.policies import (
    AlmgrenChrissPolicy,
    ConstantParticipationPolicy,
    RecalibratedAlmgrenChrissPolicy,
)
from neural_optimal_execution.policies.almgren_chriss import compute_ac_schedule


def test_ac_schedule_sells_full_inventory():
    schedule = compute_ac_schedule(
        inventory=1000.0,
        n_steps=10,
        horizon=1.0,
        volatility=0.02,
        temporary_impact=2e-7,
        risk_aversion=5e-4,
    )
    assert len(schedule) == 10
    assert np.all(schedule >= 0.0)
    assert np.isclose(schedule.sum(), 1000.0)


def test_ac_schedule_liquidates_more_in_first_half_when_risk_aversion_increases():
    slow_schedule = compute_ac_schedule(
        inventory=1000.0,
        n_steps=20,
        horizon=1.0,
        volatility=0.02,
        temporary_impact=2e-7,
        risk_aversion=1e-5,
    )
    fast_schedule = compute_ac_schedule(
        inventory=1000.0,
        n_steps=20,
        horizon=1.0,
        volatility=0.02,
        temporary_impact=2e-7,
        risk_aversion=8e-3,
    )

    slow_first_half = slow_schedule[:10].sum() / 1000.0
    fast_first_half = fast_schedule[:10].sum() / 1000.0

    assert fast_first_half > slow_first_half


def market_path_with_future_values(config, future_volatility, future_impact):
    return MarketPath(
        expected_volume=np.full(config.n_steps, config.base_daily_volume / config.n_steps),
        volume=np.full(config.n_steps, config.base_daily_volume / config.n_steps),
        volatility=np.array([config.base_volatility, *future_volatility], dtype=float),
        temp_impact=np.array([config.temp_impact, *future_impact], dtype=float),
        returns=np.zeros(config.n_steps),
        transient_noise=np.zeros(config.n_steps),
        regimes=np.zeros(config.n_steps, dtype=np.int64),
    )


def test_static_ac_schedule_does_not_depend_on_future_realized_market_states():
    config = ExecutionConfig(parent_order=1000.0, n_steps=4)
    env = ExecutionEnv(config)
    policy = AlmgrenChrissPolicy()

    env.market_path = market_path_with_future_values(config, [0.01, 0.02, 0.03], [1e-8, 2e-8, 3e-8])
    policy.reset(env)
    first_schedule = policy.schedule.copy()

    env.market_path = market_path_with_future_values(config, [0.50, 0.60, 0.70], [1e-3, 2e-3, 3e-3])
    policy.reset(env)

    assert np.allclose(policy.schedule, first_schedule)


def test_recalibrated_ac_current_action_does_not_depend_on_future_realized_states():
    config = ExecutionConfig(parent_order=1000.0, n_steps=4)
    env = ExecutionEnv(config)
    policy = RecalibratedAlmgrenChrissPolicy()
    state = env.observe()

    env.market_path = market_path_with_future_values(config, [0.01, 0.02, 0.03], [1e-8, 2e-8, 3e-8])
    first_action = policy.act(state, env)
    env.market_path = market_path_with_future_values(config, [0.50, 0.60, 0.70], [1e-3, 2e-3, 3e-3])
    second_action = policy.act(state, env)

    assert np.isclose(first_action, second_action)


def test_constant_participation_policy_uses_expected_parent_order_rate():
    config = ExecutionConfig(
        parent_order=50.0,
        n_steps=2,
        base_daily_volume=1000.0,
        participation_rate=0.10,
    )
    env = ExecutionEnv(config)
    env.market_path = market_path_with_future_values(config, [0.02], [config.temp_impact])
    env.market_path.volume[:] = 400.0

    action = ConstantParticipationPolicy().act(env.observe(), env)

    assert action == 20.0

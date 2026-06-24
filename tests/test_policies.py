import numpy as np

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

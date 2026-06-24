import importlib.util
from pathlib import Path

import numpy as np

from neural_optimal_execution.config import ExecutionConfig


def load_stress_module():
    script_path = Path(__file__).resolve().parents[1] / "experiments" / "run_stress_tests.py"
    spec = importlib.util.spec_from_file_location("run_stress_tests", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_stress_scenarios_include_required_names():
    module = load_stress_module()
    scenarios = module.build_scenarios(ExecutionConfig(n_steps=6))

    assert tuple(scenarios) == module.SCENARIO_ORDER


def test_late_day_liquidity_collapse_cuts_final_third_volume():
    module = load_stress_module()
    config = ExecutionConfig(n_steps=6, base_daily_volume=6000.0)

    stressed = module.late_day_liquidity_collapse(config)
    base_expected = config.base_daily_volume * module.base_volume_curve(config)
    stressed_expected = stressed.base_daily_volume * np.asarray(stressed.volume_curve)

    assert np.allclose(stressed_expected[:4], base_expected[:4])
    assert np.allclose(stressed_expected[4:], base_expected[4:] * 0.25)
    assert stressed.base_daily_volume < config.base_daily_volume


def test_stress_summary_uses_requested_metric_names():
    module = load_stress_module()
    losses = np.array([1.0, 2.0, 3.0, 100.0])
    terminal_inventory = np.array([0.0, 10.0])
    violations = np.array([0.0, 4.0])
    forced = np.array([0.0, 8.0])

    row = module.summarize_evaluation("base", "TWAP", losses, terminal_inventory, violations, forced, parent_order=20.0)

    assert row["scenario"] == "base"
    assert row["policy"] == "TWAP"
    assert row["mean_shortfall_bps"] == float(losses.mean())
    assert row["std_shortfall_bps"] == float(losses.std(ddof=1))
    assert row["cvar_95_bps"] == 100.0
    assert row["p99_shortfall_bps"] == float(np.quantile(losses, 0.99))
    assert row["worst_shortfall_bps"] == 100.0
    assert row["completion_rate"] == 0.5
    assert row["avg_terminal_inventory"] == 5.0
    assert row["avg_participation_violation_shares"] == 2.0
    assert row["forced_terminal_liquidation_shares"] == 4.0
    assert row["terminal_inventory_fraction"] == 0.25
    assert row["participation_violation_fraction_of_parent_order"] == 0.1
    assert row["forced_terminal_liquidation_fraction"] == 0.2

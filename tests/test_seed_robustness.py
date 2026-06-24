import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


def load_seed_robustness_module():
    script_path = Path(__file__).resolve().parents[1] / "experiments" / "run_seed_robustness.py"
    spec = importlib.util.spec_from_file_location("run_seed_robustness", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_seed_summary_uses_requested_metric_names():
    module = load_seed_robustness_module()
    losses = np.array([1.0, 2.0, 3.0, 100.0])
    terminal_inventory = np.array([0.0, 10.0])
    violations = np.array([0.0, 4.0])
    forced = np.array([0.0, 8.0])

    row = module.summarize_seed("Policy", 0, losses, terminal_inventory, violations, forced, parent_order=20.0)

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


def test_seed_aggregate_includes_mean_and_std_for_every_metric():
    module = load_seed_robustness_module()
    rows = []
    for seed, mean_shortfall in [(0, 10.0), (1, 14.0)]:
        rows.append(
            {
                "seed": seed,
                "policy": "TWAP",
                "mean_shortfall_bps": mean_shortfall,
                "std_shortfall_bps": 2.0,
                "cvar_95_bps": 20.0,
                "p99_shortfall_bps": 30.0,
                "worst_shortfall_bps": 40.0,
                "completion_rate": 1.0,
                "avg_terminal_inventory": 0.0,
                "avg_participation_violation_shares": 0.0,
                "forced_terminal_liquidation_shares": 0.0,
                "terminal_inventory_fraction": 0.0,
                "forced_terminal_liquidation_fraction": 0.0,
                "participation_violation_fraction_of_parent_order": 0.0,
            }
        )

    aggregate = module.aggregate_seed_metrics(pd.DataFrame(rows), ["TWAP"])

    assert aggregate.loc[0, "policy"] == "TWAP"
    assert aggregate.loc[0, "n_seeds"] == 2
    for metric in module.METRICS:
        assert f"{metric}_mean" in aggregate.columns
        assert f"{metric}_std" in aggregate.columns
    assert aggregate.loc[0, "mean_shortfall_bps_mean"] == 12.0
    assert np.isclose(aggregate.loc[0, "mean_shortfall_bps_std"], np.sqrt(8.0))

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

from neural_optimal_execution.config import EvaluationConfig, ExecutionConfig, load_project_config


def load_objective_comparison_module():
    script_path = Path(__file__).resolve().parents[1] / "experiments" / "run_neural_objective_comparison.py"
    spec = importlib.util.spec_from_file_location("run_neural_objective_comparison", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_objective_comparison_rejects_different_environments():
    module = load_objective_comparison_module()

    with pytest.raises(ValueError, match="identical environment"):
        module.validate_comparable_configs(
            ExecutionConfig(parent_order=100.0),
            ExecutionConfig(parent_order=200.0),
            EvaluationConfig(),
            EvaluationConfig(),
        )


def test_objective_configs_change_objective_not_environment():
    root = Path(__file__).resolve().parents[1]
    mean_env, mean_eval, mean_train, _ = load_project_config(root / "configs/objectives/mean_variance.yaml")
    cvar_env, cvar_eval, cvar_train, _ = load_project_config(root / "configs/objectives/cvar.yaml")

    assert mean_env == cvar_env
    assert mean_eval == cvar_eval
    assert mean_train.alpha_cvar == 0.0
    assert cvar_train.alpha_cvar > 0.0


def test_objective_aggregation_produces_mean_and_std_columns():
    module = load_objective_comparison_module()
    raw = pd.DataFrame(
        [
            {"training_seed": 0, "objective": "cvar", "policy": "Neural CVaR", "cvar_95_bps": 10.0},
            {"training_seed": 1, "objective": "cvar", "policy": "Neural CVaR", "cvar_95_bps": 14.0},
        ]
    )

    aggregate = module.aggregate_objective_metrics(raw)

    assert aggregate.loc[0, "cvar_95_bps_mean"] == 12.0
    assert aggregate.loc[0, "cvar_95_bps_std"] > 0.0

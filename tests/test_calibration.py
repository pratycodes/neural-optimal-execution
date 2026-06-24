import importlib.util
import sys
from pathlib import Path

import numpy as np

from neural_optimal_execution.config import load_project_config
from neural_optimal_execution.data.calibration import (
    apply_order_sizing,
    calibrate_intraday_csv,
    calibration_sufficiency_warnings,
    write_calibrated_config,
)
from neural_optimal_execution.environment.market_simulator import MarketSimulator


FIXTURE = Path(__file__).resolve().parent / "fixtures" / "tiny_intraday.csv"


def load_calibration_script():
    script_path = Path(__file__).resolve().parents[1] / "experiments" / "calibrate_from_intraday.py"
    spec = importlib.util.spec_from_file_location("calibrate_from_intraday", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_intraday_calibration_estimates_profiles_and_spread():
    calibration = calibrate_intraday_csv(FIXTURE, symbol="AAPL", bar_minutes=5)

    assert calibration.has_spread
    assert len(calibration.summary) == 4
    assert np.isclose(calibration.summary["normalized_volume"].sum(), 1.0)
    assert (calibration.summary["avg_volume"] > 0.0).all()
    assert (calibration.summary["realized_volatility"] >= 0.0).all()
    assert "avg_spread" in calibration.summary
    assert len(calibration.config["environment"]["volume_curve"]) == 4
    assert len(calibration.config["environment"]["volatility_curve"]) == 4
    assert len(calibration.config["calibration"]["liquidity_regime_labels"]) == 4


def test_calibrated_yaml_loads_and_market_simulator_uses_empirical_curves(tmp_path):
    calibration = calibrate_intraday_csv(FIXTURE, symbol="AAPL", bar_minutes=5)
    apply_order_sizing(
        calibration,
        target_order_participation=0.05,
        max_participation_rate=0.10,
    )
    output_config = tmp_path / "AAPL_5min.yaml"
    write_calibrated_config(calibration.config, output_config)

    env_cfg, _eval_cfg, _train_cfg, _ac_cfg = load_project_config(output_config)
    simulator = MarketSimulator(env_cfg)

    assert env_cfg.n_steps == 4
    assert np.isclose(env_cfg.parent_order, 362.5)
    assert np.isclose(env_cfg.participation_rate, 0.10)
    assert np.allclose(simulator.expected_volume_curve, np.asarray(env_cfg.volume_curve, dtype=float))
    assert np.allclose(simulator.expected_volatility_curve, np.asarray(env_cfg.volatility_curve, dtype=float))


def test_missing_input_message_points_to_fixture_smoke_test():
    module = load_calibration_script()
    message = module.missing_input_message("data/raw/sample_intraday.csv")

    assert "Input CSV not found" in message
    assert "--input tests/fixtures/tiny_intraday.csv" in message
    assert "--output-config configs/calibrated/AAPL_5min.yaml" in message


def test_calibration_sufficiency_warnings_on_tiny_sample():
    calibration = calibrate_intraday_csv(FIXTURE, symbol="AAPL", bar_minutes=5)

    warnings = calibration_sufficiency_warnings(calibration)

    assert any("fewer than 20 time buckets" in warning for warning in warnings)
    assert any("fewer than 5 trading days" in warning for warning in warnings)
    assert any("bid/ask spread is constant" in warning for warning in warnings)
    assert any("volume curve has fewer than 20 points" in warning for warning in warnings)


def test_calibration_script_prints_tiny_sample_warnings(tmp_path, monkeypatch, capsys):
    module = load_calibration_script()
    output_config = tmp_path / "AAPL_5min.yaml"
    output_dir = tmp_path / "results"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "calibrate_from_intraday.py",
            "--input",
            str(FIXTURE),
            "--symbol",
            "AAPL",
            "--bar-minutes",
            "5",
            "--output-config",
            str(output_config),
            "--output-dir",
            str(output_dir),
        ],
    )

    exit_code = module.main()
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "WARNING: fewer than 20 time buckets" in captured.err
    assert output_config.exists()


def test_auto_sized_parent_order_is_feasible_and_uses_actual_share_volume():
    calibration = calibrate_intraday_csv(FIXTURE, symbol="AAPL", bar_minutes=5)

    sizing = apply_order_sizing(
        calibration,
        target_order_participation=0.05,
        max_participation_rate=0.10,
    )

    assert np.isclose(sizing.total_expected_volume_over_horizon, 7250.0)
    assert np.isclose(sizing.recommended_parent_order_size, 362.5)
    assert np.isclose(sizing.configured_parent_order_size, 362.5)
    assert np.isclose(sizing.feasible_max_order_size, 580.0)
    assert sizing.is_participation_feasible
    assert np.isclose(calibration.config["environment"]["parent_order"], 362.5)
    assert np.isclose(calibration.summary["configured_parent_order_size"].iloc[0], 362.5)
    assert np.isclose(calibration.summary["normalized_volume"].sum(), 1.0)


def test_manually_feasible_parent_order_passes():
    calibration = calibrate_intraday_csv(FIXTURE, symbol="AAPL", bar_minutes=5)

    sizing = apply_order_sizing(
        calibration,
        target_order_participation=0.05,
        max_participation_rate=0.10,
        parent_order_size=500.0,
    )

    assert sizing.is_participation_feasible
    assert np.isclose(sizing.configured_parent_order_size, 500.0)
    assert np.isclose(calibration.config["environment"]["parent_order"], 500.0)
    assert np.isclose(calibration.config["environment"]["participation_rate"], 0.10)


def test_auto_resize_caps_manually_oversized_parent_order():
    calibration = calibrate_intraday_csv(FIXTURE, symbol="AAPL", bar_minutes=5)

    sizing = apply_order_sizing(
        calibration,
        target_order_participation=0.05,
        max_participation_rate=0.10,
        parent_order_size=1000.0,
        auto_resize_order=True,
    )

    assert sizing.was_auto_resized
    assert sizing.is_participation_feasible
    assert np.isclose(sizing.configured_parent_order_size, 580.0)
    assert np.isclose(calibration.config["environment"]["parent_order"], 580.0)


def test_manually_oversized_parent_order_fails_without_auto_resize(tmp_path, monkeypatch, capsys):
    module = load_calibration_script()
    output_config = tmp_path / "oversized.yaml"
    output_dir = tmp_path / "results"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "calibrate_from_intraday.py",
            "--input",
            str(FIXTURE),
            "--symbol",
            "AAPL",
            "--bar-minutes",
            "5",
            "--parent-order-size",
            "1000",
            "--output-config",
            str(output_config),
            "--output-dir",
            str(output_dir),
        ],
    )

    exit_code = module.main()
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "Requested calibrated parent order is too large" in captured.err
    assert "feasible maximum order size: 580.00 shares" in captured.err
    assert not output_config.exists()

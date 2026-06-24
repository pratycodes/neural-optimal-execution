import numpy as np

from neural_optimal_execution.config import ExecutionConfig
from neural_optimal_execution.evaluation.feasibility import participation_feasibility
from neural_optimal_execution.evaluation.outputs import make_run_output_dirs, run_output_root


def test_infeasible_order_size_detection():
    config = ExecutionConfig(
        parent_order=200.0,
        n_steps=4,
        base_daily_volume=1000.0,
        participation_rate=0.10,
    )

    report = participation_feasibility(config)

    assert not report.feasible
    assert np.isclose(report.total_participation_capacity, 100.0)
    assert np.isclose(report.required_participation_rate, 0.20)


def test_feasible_order_size_passes():
    config = ExecutionConfig(
        parent_order=50.0,
        n_steps=4,
        base_daily_volume=1000.0,
        participation_rate=0.10,
    )

    report = participation_feasibility(config)

    assert report.feasible
    assert np.isclose(report.total_participation_capacity, 100.0)


def test_feasibility_uses_empirical_volume_curve_as_share_profile():
    config = ExecutionConfig(
        parent_order=100.0,
        n_steps=4,
        base_daily_volume=1000.0,
        participation_rate=0.10,
        volume_curve=(0.10, 0.20, 0.30, 0.40),
    )

    report = participation_feasibility(config)

    assert report.feasible
    assert np.isclose(report.total_expected_volume, 1000.0)
    assert np.isclose(report.total_participation_capacity, 100.0)


def test_run_specific_output_directory_creation(tmp_path):
    dirs = make_run_output_dirs(tmp_path / "results", "AAPL_5min_calibrated")

    assert dirs.root == tmp_path / "results" / "runs" / "AAPL_5min_calibrated"
    assert dirs.tables.is_dir()
    assert dirs.figures.is_dir()
    assert dirs.models.is_dir()


def test_omitted_run_name_keeps_backward_compatible_output_root(tmp_path):
    assert run_output_root(tmp_path / "results") == tmp_path / "results"

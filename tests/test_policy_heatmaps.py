import importlib.util
from pathlib import Path

from neural_optimal_execution.config import ExecutionConfig


def load_policy_heatmaps_module():
    script_path = Path(__file__).resolve().parents[1] / "experiments" / "plot_policy_heatmaps.py"
    spec = importlib.util.spec_from_file_location("plot_policy_heatmaps", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_heatmap_baseline_state_uses_economic_neutral_values():
    module = load_policy_heatmaps_module()
    state = module.baseline_states(1, ExecutionConfig())[0]

    assert state[0] == 0.50
    assert state[1] == 0.50
    assert state[2] == 0.50
    assert state[3] == 0.00
    assert state[4] == 1.00
    assert state[5] == 1.00
    assert state[6] == 1.00
    assert state[7] == 1.00
    assert state[8] == 0.00


def test_missing_model_message_points_to_training_command():
    module = load_policy_heatmaps_module()
    message = module.missing_model_message("results/models/neural_policy.pt", "configs/default.yaml")

    assert "Saved neural policy not found" in message
    assert "python experiments/train_neural_policy.py --config configs/default.yaml" in message

.PHONY: install test baselines train compare clean-results

install:
	python -m pip install -r requirements.txt
	python -m pip install -e .

test:
	pytest -q

baselines:
	python experiments/run_baselines.py --config configs/default.yaml

train:
	python experiments/train_neural_policy.py --config configs/default.yaml

compare:
	python experiments/run_full_comparison.py --config configs/default.yaml

clean-results:
	rm -f results/tables/*.csv results/figures/*.png results/models/*.pt

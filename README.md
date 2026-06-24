\# Neural Optimal Execution

A research framework for studying optimal execution under market impact, stochastic liquidity, participation constraints, and tail-risk-aware control.

The goal is not to predict markets or generate trading signals. Instead, this project focuses on a fundamental institutional trading problem:

> Given a large order that must be executed over a fixed horizon, how should it be split through time to minimize execution cost and control downside risk?

The framework combines classical execution models, empirical market calibration, differentiable simulation, and neural control policies within a unified experimental environment.

---

# Highlights

### Market Environment

* Stochastic liquidity regimes
* Intraday volume seasonality
* Intraday volatility seasonality
* Temporary market impact
* Permanent market impact
* Transient impact decay
* Participation-rate constraints
* Forced-completion and strict-execution modes

### Execution Policies

* TWAP
* VWAP
* Constant Participation
* Almgren-Chriss
* Recalibrated Almgren-Chriss
* Neural Execution Policies

### Research Infrastructure

* Empirical calibration from intraday market data
* Differentiable PyTorch simulator
* Multi-seed robustness testing
* Stress testing framework
* Policy diagnostics and adaptivity analysis
* Automated experiment tracking

---

# Key Research Findings

## 1. Objective Design Matters More Than Model Complexity

One of the strongest findings from this project is that the training objective had a significantly larger impact on performance than architectural changes.

### Neural Objective Comparison

| Objective     | Mean Shortfall (bps) | Std (bps) | CVaR95 (bps) |  P99 (bps) | Worst (bps) |
| ------------- | -------------------: | --------: | -----------: | ---------: | ----------: |
| Mean-Variance |                -4.60 |     75.96 |       151.30 |     168.59 |      215.04 |
| CVaR          |                -4.60 | **64.84** |   **128.96** | **141.26** |  **185.38** |

### Conclusion

* Both objectives achieve nearly identical average execution performance.
* Explicit CVaR optimization reduces tail risk substantially.
* Improvements are visible across every downside-risk metric.
* Tail-aware optimization is a more impactful research direction than larger neural architectures.

---

## 2. Classical Baselines Remain Competitive

The calibrated execution environment produced an important result:

### Calibrated AAPL 5-Minute Environment

| Policy                 | Mean Shortfall (bps) | Std (bps) | CVaR95 (bps) |  P99 (bps) |
| ---------------------- | -------------------: | --------: | -----------: | ---------: |
| TWAP                   |            **0.235** |     92.45 |       200.04 |     218.56 |
| Almgren-Chriss         |                0.247 |     91.23 |       197.39 |     214.55 |
| Recalibrated AC        |                0.274 |     88.78 |       191.48 |     210.03 |
| VWAP                   |                0.381 |     80.86 |       172.87 |     184.48 |
| Neural Policy          |                0.631 |     79.54 |       168.68 |     182.38 |
| Constant Participation |                0.694 | **78.69** |   **166.88** | **180.30** |

### Conclusion

* Neural policies reduce dispersion and tail risk.
* Classical execution schedules remain difficult to outperform.
* Simulator calibration changes policy rankings materially.
* The next performance gains are likely to come from richer market-state information rather than deeper models.

---

## 3. Architecture Complexity Did Not Improve Results

A recurrent GRU-based execution policy was implemented and evaluated against the original feedforward policy.

### Outcome

| Architecture    | Result                    |
| --------------- | ------------------------- |
| Feedforward MLP | Strong baseline           |
| GRU             | No meaningful improvement |

### Conclusion

The current execution state already contains sufficient information regarding:

* Inventory
* Time remaining
* Volatility
* Volume
* Expected liquidity
* Market impact state

Adding memory produced negligible gains, suggesting that the execution problem is approximately Markovian under the current simulator design.

---

# Research Contributions

### Execution Simulator

Built a configurable execution environment incorporating:

* Market impact
* Liquidity uncertainty
* Participation constraints
* Intraday seasonality
* Tail-risk evaluation

### Empirical Calibration Pipeline

Developed a calibration workflow that converts intraday market data into executable simulation environments while preserving reproducibility.

### Benchmark Suite

Implemented and evaluated:

* TWAP
* VWAP
* Constant Participation
* Almgren-Chriss
* Recalibrated Almgren-Chriss
* Neural Control Policies

### Risk-Aware Optimization

Demonstrated that:

* CVaR optimization consistently improves tail-risk outcomes.
* Calibration materially affects benchmark rankings.
* Architectural complexity alone does not guarantee better execution quality.

---

# Installation

```bash
git clone <repo-url>
cd neural-optimal-execution

python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
pip install -e .
```

---

# Running Experiments

## Train Neural Policy

```bash
python experiments/train_neural_policy.py \
    --config configs/default.yaml
```

## Full Policy Comparison

```bash
python experiments/run_full_comparison.py \
    --config configs/default.yaml \
    --run-name experiment_1
```

## Empirical Calibration

```bash
python experiments/calibrate_from_intraday.py \
    --input data/raw/sample_intraday.csv \
    --symbol AAPL \
    --bar-minutes 5 \
    --target-order-participation 0.05 \
    --max-participation-rate 0.10 \
    --output-config configs/calibrated/AAPL_5min.yaml
```

## Calibrated Comparison

```bash
python experiments/run_full_comparison.py \
    --config configs/calibrated/AAPL_5min.yaml \
    --run-name AAPL_5min
```

## Objective Comparison

```bash
python experiments/run_neural_objective_comparison.py
```

## Stress Testing

```bash
python experiments/run_stress_tests.py \
    --config configs/default.yaml
```

---

# Current Limitations

The current framework does not yet model:

* Bid-ask spread dynamics
* Order-book depth
* Order imbalance
* Queue position effects
* Cross-asset execution
* Data-driven impact estimation

These represent the primary directions for future work.

---

# Future Work

Priority roadmap:

1. Incorporate market microstructure features:

   * spread
   * depth
   * order imbalance
   * microprice

2. Replace hand-specified impact parameters with learned impact models.

3. Extend to multi-asset execution.

4. Calibrate using full limit-order-book datasets.

5. Investigate distributionally robust execution objectives.

---

# Disclaimer

This repository is intended for research and educational purposes only. It is not a live trading system, investment product, or execution engine. Results are simulator-based and should not be interpreted as real-world trading performance.

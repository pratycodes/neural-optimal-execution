# Neural Optimal Execution

Market impact, stochastic liquidity, and CVaR-aware execution policies.

This repository is a research framework for the optimal execution problem: how to split a known parent order over a fixed horizon while accounting for market impact, volatility risk, stochastic liquidity, transient impact, and tail execution risk.

The project is deliberately framed as **optimal execution**, not as a trading bot. The model does not decide what asset to buy or sell. It decides how to execute a known large order.

## Research question

Can a neural stochastic-control policy reduce implementation shortfall and tail execution losses compared with TWAP, VWAP, and Almgren-Chriss style baselines under stochastic liquidity and transient market impact?

## Current MVP

Implemented in this starter repo:

- Synthetic single-asset execution environment for sell-side liquidation
- Stochastic volatility and stochastic liquidity regimes
- Temporary impact, permanent impact, and transient impact state
- Participation-rate constraint
- TWAP baseline
- VWAP baseline
- Static Almgren-Chriss style baseline
- Recalibrated Almgren-Chriss style baseline
- Implementation shortfall, standard deviation, CVaR, p99/worst-tail cost, completion rate, and participation-violation metrics
- Direct neural policy optimization through a differentiable PyTorch simulator
- Config-driven experiment scripts
- Baseline and full-comparison plots and result tables

## Repository structure

```text
neural-optimal-execution/
  README.md
  PRD.md
  pyproject.toml
  requirements.txt
  Makefile
  configs/
    default.yaml
    stress_liquidity.yaml
  experiments/
    run_baselines.py
    train_neural_policy.py
    run_full_comparison.py
    run_seed_robustness.py
  src/neural_optimal_execution/
    __init__.py
    config.py
    environment/
      __init__.py
      execution_env.py
      market_simulator.py
    policies/
      __init__.py
      base.py
      twap.py
      vwap.py
      almgren_chriss.py
      neural_policy.py
    training/
      __init__.py
      losses.py
      train_policy.py
    evaluation/
      __init__.py
      metrics.py
      plots.py
      stress_tests.py
  tests/
    test_environment.py
    test_metrics.py
    test_neural_policy.py
    test_policies.py
  notebooks/
    .gitkeep
  report/
    README.md
  results/
    figures/
    tables/
    models/
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

`pyproject.toml` defines the package and runtime dependencies. `requirements.txt` installs those runtime dependencies plus `pytest` for local test runs.

## Make targets

```bash
make install
make test
make baselines
make train
make compare
make clean-results
```

The experiment targets are aliases for the explicit commands shown below.

## Run tests

```bash
pytest -q
```

## Run classical baselines

```bash
python experiments/run_baselines.py --config configs/default.yaml
```

This evaluates TWAP, VWAP, Almgren-Chriss, and recalibrated Almgren-Chriss over Monte Carlo execution episodes.

Outputs:

```text
results/tables/baseline_metrics.csv
results/figures/cost_distributions.png
results/figures/average_inventory.png
```

## Train the neural execution policy

```bash
python experiments/train_neural_policy.py --config configs/default.yaml
```

The neural policy observes normalized execution state variables and outputs a fraction of the maximum allowable trade size:

```text
u_t = min(q_t, participation_rate * V_t) * sigmoid(f_theta(s_t))
```

The training objective combines normalized implementation shortfall, variance, optional CVaR, and terminal inventory penalty.

## Run full comparison

After training, compare the neural policy against all classical baselines:

```bash
python experiments/run_full_comparison.py --config configs/default.yaml
```

Or train and evaluate in one command:

```bash
python experiments/run_full_comparison.py --config configs/default.yaml --retrain
```

Outputs:

```text
results/tables/full_comparison_metrics.csv
results/figures/full_cost_distributions.png
results/figures/full_average_inventory.png
```

## Multi-Seed Robustness

Evaluate the saved policies across five random test-environment seeds:

```bash
python experiments/run_seed_robustness.py --config configs/default.yaml
```

The script evaluates TWAP, VWAP, Almgren-Chriss, Recalibrated AC, and Neural Policy when `results/models/neural_policy.pt` exists. It does not retrain the neural policy.

Outputs:

```text
results/tables/seed_robustness_raw.csv
results/tables/seed_robustness_metrics.csv
results/figures/seed_robustness_cvar.png
results/figures/seed_robustness_mean_shortfall.png
```

On the current synthetic default environment and saved checkpoint, across 5 seeds, Neural Policy achieved:

```text
mean_shortfall_bps_mean = 1.62
std_shortfall_bps_mean = 84.56
cvar_95_bps_mean = 181.88
p99_shortfall_bps_mean = 208.11
worst_shortfall_bps_mean = 290.98
completion_rate_mean = 1.0
avg_terminal_inventory_mean = 0.0
avg_participation_violation_shares_mean = 0.0
```

Against Recalibrated AC in this synthetic test, the neural policy had about 18.8% lower mean shortfall, 32.6% lower cost standard deviation, 29.6% lower CVaR 95, 28.1% lower p99 shortfall, and 25.3% lower worst shortfall. These results are evidence for this simulator configuration only; they should not be read as live-market or empirically calibrated execution performance.

## Core model

State:

```text
s_t = (q_t, time_remaining, time_elapsed, S_t, sigma_t, V_t, expected_volume_t, eta_t, Y_t)
```

Action:

```text
u_t = shares sold at time t
```

Execution price for sell-side liquidation:

```text
P_exec_t = S_t - eta_t * u_t - Y_t
```

Transient impact state:

```text
Y_{t+1} = rho * Y_t + phi * u_t + epsilon_t
```

Implementation shortfall:

```text
IS = X_0 S_0 - sum_t u_t P_exec_t
```

## Resume title

**Neural Optimal Execution under Market Impact and Liquidity Risk**

Suggested resume bullets:

- Built a stochastic optimal-execution framework comparing TWAP, VWAP, Almgren-Chriss, dynamic recalibration, and neural control policies under temporary, permanent, and transient market impact.
- Modeled stochastic volatility, intraday volume, liquidity regimes, and participation-rate constraints; trained risk-aware neural policies to minimize implementation shortfall, variance, CVaR, and terminal inventory penalties.
- Evaluated policies through execution-cost distributions, stress tests, and learned policy diagnostics under liquidity shocks and impact-parameter misspecification.

## Next build steps

1. Add policy heatmaps for interpretability.
2. Add a stress-test experiment runner around `src/neural_optimal_execution/evaluation/stress_tests.py`.
3. Add CVaR-aware training config and comparison table.
4. Write the research report in `report/`.

## Disclaimer

This repository is for research and education. It is not a live trading system, broker integration, or investment recommendation engine.

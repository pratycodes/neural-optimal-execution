# Neural Optimal Execution

Market impact, stochastic liquidity, and CVaR-aware execution policies.

This repository is a research framework for optimal execution: how to split a known parent sell order over a fixed horizon while accounting for market impact, volatility risk, stochastic liquidity, transient impact, participation constraints, and tail execution risk.

The project is deliberately framed as **optimal execution**, not as a trading bot. The model does not decide what asset to buy or sell. It decides how to execute a known large order.

## Research Question

Can a neural stochastic-control policy reduce implementation shortfall and tail execution losses compared with TWAP, VWAP, and Almgren-Chriss style baselines under stochastic liquidity, transient market impact, empirical intraday profiles, and strict participation constraints?

## Current MVP

Implemented in this repo:

- Synthetic single-asset sell-side execution environment
- Empirical intraday calibration from CSV or yfinance-downloaded bars
- Stochastic volatility and stochastic liquidity regimes
- Temporary impact, permanent impact, and transient impact state
- Participation-rate constraints with both forced-completion and strict modes
- TWAP, VWAP, static Almgren-Chriss, and recalibrated Almgren-Chriss baselines
- Direct neural policy optimization through a differentiable PyTorch simulator
- Implementation shortfall, standard deviation, CVaR, p99/worst-tail cost, completion, terminal inventory, forced-liquidation, and participation-violation metrics
- Multi-seed robustness, stress tests, Almgren-Chriss sanity checks, and policy heatmap diagnostics
- Run-specific output directories to avoid mixing default and calibrated artifacts

## Repository Structure

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
    calibrated/
      AAPL_5min.yaml
  data/
    README.md
    raw/                  # git-ignored raw market data
  experiments/
    download_yfinance_intraday.py
    calibrate_from_intraday.py
    run_baselines.py
    train_neural_policy.py
    run_full_comparison.py
    run_seed_robustness.py
    run_stress_tests.py
    run_ac_sanity_check.py
    plot_policy_heatmaps.py
  src/neural_optimal_execution/
    config.py
    data/
      calibration.py
    environment/
      execution_env.py
      market_simulator.py
    policies/
      twap.py
      vwap.py
      almgren_chriss.py
      neural_policy.py
    training/
      losses.py
      train_policy.py
    evaluation/
      feasibility.py
      metrics.py
      outputs.py
      plots.py
      stress_tests.py
  tests/
  results/
    tables/
    figures/
    models/
    runs/
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

`requirements.txt` includes `yfinance` for optional intraday data download and `pytest` for local verification.

## Make Targets

```bash
make install
make test
make baselines
make train
make compare
make clean-results
```

The experiment targets are aliases for the default synthetic commands.

## Run Tests

```bash
pytest -q
```

Current verification: `51 passed`.

## Default Commands

These commands keep the original backward-compatible output locations under `results/`:

```bash
pytest -q
python experiments/run_baselines.py --config configs/default.yaml
python experiments/train_neural_policy.py --config configs/default.yaml
python experiments/run_full_comparison.py --config configs/default.yaml
python experiments/run_seed_robustness.py --config configs/default.yaml
python experiments/run_stress_tests.py --config configs/default.yaml
```

For research runs, prefer adding `--run-name` as shown below.

The baseline-only command writes `results/tables/baseline_metrics.csv`, `results/figures/cost_distributions.png`, and `results/figures/average_inventory.png`. The named full-comparison runs are the better source for current policy comparisons.

## Core Model

State:

```text
s_t = (q_t, time_remaining, time_elapsed, S_t, sigma_t, V_t, expected_volume_t, eta_t, Y_t)
```

Action:

```text
u_t = shares sold at time t
```

Neural policy action:

```text
u_t = min(q_t, participation_rate * V_t) * sigmoid(f_theta(s_t))
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
shortfall_bps = 10000 * IS / (X_0 S_0)
```

Lower positive shortfall is better. Negative shortfall can occur when realized price drift/noise is favorable for a sell order.

## Output Directories

Use `--run-name` for research runs so default and calibrated artifacts do not overwrite each other:

```text
results/runs/<run_name>/tables/
results/runs/<run_name>/figures/
results/runs/<run_name>/models/
```

If `--run-name` is omitted, scripts keep backward-compatible outputs under `results/tables/`, `results/figures/`, and `results/models/`.

## Constraint Modes

Default forced-completion mode keeps legacy behavior:

- policies are clipped by inventory and participation limits before the final step
- terminal liquidation can force completion on the last step
- forced shares are reported as `forced_terminal_liquidation_shares`

Strict mode is enabled with `--strict-constraints`:

- every action is clipped to `min(q_t, participation_rate * V_t)`
- terminal forced liquidation is disabled
- leftover inventory remains as terminal inventory
- `forced_terminal_liquidation_shares` is exactly zero
- completion rate reflects whether terminal inventory is effectively zero

## Synthetic Default Run

Train and evaluate the default synthetic setup:

```bash
python experiments/train_neural_policy.py \
  --config configs/default.yaml \
  --run-name default_synthetic

python experiments/run_full_comparison.py \
  --config configs/default.yaml \
  --run-name default_synthetic

python experiments/run_seed_robustness.py \
  --config configs/default.yaml \
  --run-name default_synthetic

python experiments/run_stress_tests.py \
  --config configs/default.yaml \
  --run-name default_synthetic
```

Key current artifacts:

```text
results/runs/default_synthetic/tables/full_comparison_metrics.csv
results/runs/default_synthetic/tables/seed_robustness_metrics.csv
results/runs/default_synthetic/tables/stress_test_metrics.csv
```

Current five-seed synthetic summary:

| Policy | Mean bps | Std bps | CVaR 95 bps | p99 bps | Worst bps | Completion |
|---|---:|---:|---:|---:|---:|---:|
| TWAP | 2.05 | 136.70 | 280.68 | 312.30 | 420.41 | 1.000 |
| VWAP | 2.07 | 131.46 | 269.44 | 300.18 | 400.50 | 1.000 |
| Almgren-Chriss | 1.94 | 128.16 | 263.56 | 295.53 | 398.61 | 1.000 |
| Recalibrated AC | 1.99 | 125.46 | 258.49 | 289.43 | 389.68 | 1.000 |
| Neural Policy | 1.62 | 84.57 | 181.89 | 208.13 | 291.01 | 1.000 |

In the synthetic default environment, Neural Policy has the lowest seed-robust tail risk and cost dispersion. It also has the lowest average CVaR in every configured stress scenario. These claims are simulator-specific.

## Empirical Calibration From Intraday Data

Raw market data belongs under `data/raw/`, which is git-ignored.

Download AAPL 5-minute bars with yfinance:

```bash
python experiments/download_yfinance_intraday.py \
  --symbol AAPL \
  --interval 5m \
  --period 60d \
  --output data/raw/sample_intraday.csv
```

Calibrate the simulator:

```bash
python experiments/calibrate_from_intraday.py \
  --input data/raw/sample_intraday.csv \
  --symbol AAPL \
  --bar-minutes 5 \
  --target-order-participation 0.05 \
  --max-participation-rate 0.10 \
  --output-config configs/calibrated/AAPL_5min.yaml
```

Current calibration summary:

| Field | Value |
|---|---:|
| Time buckets | 78 |
| Total expected horizon volume | 34,535,205 shares |
| Target order participation | 5.0% |
| Max bucket participation | 10.0% |
| Configured parent order | 1,726,760 shares |
| Feasible safety-capped max order | 2,762,816 shares |
| Participation feasible | true |

The empirical volume curve sums to 1.0 as a normalized profile, while total expected tradable volume is stored separately as actual shares. Current regime labels are mostly stressed/normal liquidity: 44 stressed, 29 normal, 5 high-liquidity buckets.

## Calibrated Forced-Completion Run

Run the empirically calibrated environment in backward-compatible forced-completion mode:

```bash
python experiments/train_neural_policy.py \
  --config configs/calibrated/AAPL_5min.yaml \
  --run-name AAPL_5min_calibrated

python experiments/run_full_comparison.py \
  --config configs/calibrated/AAPL_5min.yaml \
  --run-name AAPL_5min_calibrated

python experiments/run_seed_robustness.py \
  --config configs/calibrated/AAPL_5min.yaml \
  --run-name AAPL_5min_calibrated

python experiments/run_stress_tests.py \
  --config configs/calibrated/AAPL_5min.yaml \
  --run-name AAPL_5min_calibrated
```

Current five-seed calibrated forced-completion summary:

| Policy | Mean bps | Std bps | CVaR 95 bps | p99 bps | Worst bps | Forced shares |
|---|---:|---:|---:|---:|---:|---:|
| TWAP | 0.90 | 97.29 | 193.37 | 213.66 | 263.48 | 68.63 |
| VWAP | 1.13 | 85.22 | 170.03 | 188.17 | 231.24 | 3,516.09 |
| Almgren-Chriss | 0.85 | 95.67 | 190.02 | 210.62 | 260.01 | 68,030.59 |
| Recalibrated AC | 0.86 | 93.45 | 185.50 | 205.47 | 252.81 | 16.75 |
| Neural Policy | 1.01 | 82.93 | 165.15 | 184.28 | 234.83 | 18,064.50 |

In this forced-completion calibrated run, Neural Policy has the lowest five-seed CVaR and standard deviation, but it relies on nonzero terminal-forced shares. Forced-completion results are useful diagnostics, not the cleanest final research comparison under hard participation constraints.

Stress tests in forced-completion mode show VWAP is most robust on average across non-base calibrated stresses. Neural Policy has the lowest CVaR in volatility spike, slow transient decay, and impact misspecification, while VWAP is better under liquidity drought and late-day liquidity collapse.

## Calibrated Strict-Constraint Run

Use strict mode for final participation-constrained evaluation:

```bash
python experiments/run_full_comparison.py \
  --config configs/calibrated/AAPL_5min.yaml \
  --run-name AAPL_5min_calibrated_strict \
  --strict-constraints

python experiments/run_seed_robustness.py \
  --config configs/calibrated/AAPL_5min.yaml \
  --run-name AAPL_5min_calibrated_strict \
  --strict-constraints

python experiments/run_stress_tests.py \
  --config configs/calibrated/AAPL_5min.yaml \
  --run-name AAPL_5min_calibrated_strict \
  --strict-constraints \
  --allow-infeasible
```

Current five-seed calibrated strict summary:

| Policy | Mean bps | Std bps | CVaR 95 bps | p99 bps | Worst bps | Completion | Avg terminal shares |
|---|---:|---:|---:|---:|---:|---:|---:|
| TWAP | 1.31 | 97.21 | 193.37 | 213.66 | 263.48 | 0.997 | 68.63 |
| VWAP | 21.54 | 111.48 | 320.88 | 389.19 | 503.44 | 0.904 | 3,516.09 |
| Almgren-Chriss | 394.75 | 256.71 | 1,020.37 | 1,111.21 | 1,304.44 | 0.000 | 68,030.59 |
| Recalibrated AC | 0.96 | 93.34 | 185.50 | 205.47 | 252.81 | 0.999 | 16.75 |
| Neural Policy | 1.27 | 66.04 | 134.55 | 155.13 | 190.36 | 0.765 | 0.76 |

Strict mode changes the interpretation:

- All forced liquidation and participation-violation fractions are zero.
- Neural Policy has the lowest strict-mode tail risk and dispersion in the base calibrated seed-robust test, but its exact completion rate is lower because it often leaves tiny residual inventory.
- Recalibrated AC is the strongest classical baseline in strict base evaluation: near-complete execution, low mean shortfall, and lower tail risk than TWAP/VWAP/AC.
- Static Almgren-Chriss is not competitive in strict calibrated evaluation because it leaves material terminal inventory.
- Under severe liquidity drought and late-day collapse stress, strict completion failures are economically meaningful. VWAP has the best average non-base calibrated strict stress CVaR, while Neural Policy remains strongest in volatility and impact-oriented stresses.

## Almgren-Chriss Sanity Check

Run:

```bash
python experiments/run_ac_sanity_check.py --config configs/default.yaml
```

Outputs:

```text
results/tables/ac_sanity_check.csv
results/figures/ac_risk_aversion_inventory.png
results/figures/ac_cost_risk_frontier.png
```

Current result: higher risk aversion liquidates faster. The first-half liquidation fraction rises monotonically from 0.50 at risk aversion 0 to 0.89 at risk aversion 0.008, while CVaR falls from 276.77 bps to 160.01 bps in the synthetic setup.

## Policy Heatmaps

Run:

```bash
python experiments/plot_policy_heatmaps.py --config configs/default.yaml
```

Outputs:

```text
results/figures/policy_heatmap_inventory_time.png
results/figures/policy_heatmap_liquidity_volatility.png
results/figures/policy_heatmap_transient_impact.png
results/policy_heatmap_interpretation.md
```

Current interpretation: the saved neural policy's raw trade-fraction output is nearly flat around 0.48 across the tested local state slices. In share terms, liquidity still matters because the fraction is multiplied by the current maximum allowable trade, but the learned fraction itself is not strongly state-adaptive in these heatmaps. This is an important limitation of the current checkpoint.

## Methodology Update Workflows

The checked-in result tables predate the methodology changes below and should be regenerated before making updated research claims.

- Static Almgren-Chriss now uses config-time volatility and impact estimates rather than realized future path averages.
- Neural training and numpy evaluation use the same Markov liquidity-regime transition model.
- Strict neural execution uses an expected-future-capacity completion floor, and completion uses a parent-order-relative tolerance.
- Full comparisons include a Constant Participation baseline.

Compare mean-variance and CVaR-aware neural objectives across matched training seeds:

```bash
python experiments/run_neural_objective_comparison.py
```

Measure state-feature dependence of a saved checkpoint:

```bash
python experiments/analyze_policy_adaptivity.py \
  --config configs/default.yaml \
  --run-name default_synthetic
```

## Result Interpretation

The current result set supports these bounded conclusions:

- On the synthetic default simulator, Neural Policy reduces tail risk and volatility relative to classical baselines.
- Empirical calibration changes the problem materially. The calibrated order is feasible by construction, but liquidity shape and bucket-level constraints create policy-specific completion behavior.
- Forced-completion mode is useful for diagnosing policies, but strict mode is the more defensible setting for final participation-constrained research claims.
- Under calibrated strict constraints, Neural Policy is attractive on tail-risk metrics but needs better completion control; Recalibrated AC is the most reliable classical benchmark.
- Severe liquidity stress exposes the difference between low execution cost and guaranteed completion. Strict-mode terminal inventory should be reported alongside shortfall metrics.
- These results are research diagnostics on synthetic and yfinance-calibrated data. They are not live-market performance claims.

## Resume Title

**Neural Optimal Execution under Market Impact and Liquidity Risk**

Suggested resume bullets:

- Built a stochastic optimal-execution framework comparing TWAP, VWAP, Almgren-Chriss, dynamic recalibration, and neural control policies under temporary, permanent, and transient market impact.
- Calibrated synthetic execution simulations from intraday yfinance data while preserving a controlled simulator for stress testing and reproducible policy comparison.
- Evaluated policies through seed robustness, strict participation constraints, tail-risk metrics, stress tests, and learned policy diagnostics under liquidity shocks and impact-parameter misspecification.

## Disclaimer

This repository is for research and education. It is not a live trading system, broker integration, or investment recommendation engine. yfinance data is used for calibration experiments only; review Yahoo Finance and yfinance terms before using downloaded data outside personal research.

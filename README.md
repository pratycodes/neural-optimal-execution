# Neural Optimal Execution

A research framework for optimal execution under market impact, stochastic liquidity, participation constraints, and tail-risk-aware control.

## Key Findings

This project investigates how a large institutional order should be executed under realistic market frictions.

Unlike most execution studies that focus solely on average implementation shortfall, this framework evaluates policies using:

* Mean execution cost
* Cost volatility
* CVaR (Conditional Value at Risk)
* Extreme tail losses
* Completion risk
* Participation constraint violations

### Main Result

Across multiple experiments, objective design mattered more than model complexity.

Key observations:

* CVaR-optimized policies consistently reduced tail execution risk relative to mean-variance objectives.
* Recurrent architectures (GRU) provided little improvement over a feedforward policy, suggesting that the execution problem is approximately Markovian under the current state representation.
* Empirically calibrated liquidity profiles materially changed policy rankings relative to purely synthetic environments.
* Classical execution schedules remained highly competitive when calibrated correctly.
* Neural policies demonstrated their strongest advantage in tail-risk management rather than average execution cost minimization.

## Research Question

Given a known parent order, how should execution be scheduled to minimize implementation shortfall while controlling extreme downside risk under:

* Temporary market impact
* Permanent market impact
* Transient market impact
* Stochastic liquidity regimes
* Participation constraints
* Empirical intraday volume profiles

## Framework Components

### Classical Baselines

* TWAP
* VWAP
* Almgren-Chriss
* Dynamically recalibrated Almgren-Chriss
* Constant participation strategies

### Neural Control Policies

* Feedforward MLP execution policy
* CVaR-optimized execution policy
* Mean-variance execution policy
* Experimental recurrent (GRU) policies

### Market Simulator

The simulator includes:

* Regime-switching liquidity
* Intraday volume seasonality
* Volatility seasonality
* Temporary impact
* Permanent impact
* Transient impact decay
* Strict participation constraints
* Forced-completion diagnostics

## Current Conclusion

The strongest result from this project is not that neural networks outperform classical execution schedules.

Instead:

1. Tail-risk-aware objectives substantially improve execution robustness.
2. Simulator calibration is more important than increasing model complexity.
3. Richer market-state information is likely more valuable than deeper architectures.

Future work focuses on incorporating:

* Bid-ask spread dynamics
* Order book imbalance
* Depth-based liquidity measures
* Data-driven impact estimation
* Market microstructure calibration

rather than larger neural architectures.

## Example Results

### Objective Comparison

| Objective     | Mean Shortfall (bps) | CVaR95 (bps) |
| ------------- | -------------------- | ------------ |
| Mean-Variance | -4.60                | 151.30       |
| CVaR          | -4.60                | 128.96       |

The CVaR objective achieves materially lower tail risk while maintaining similar average execution quality.

### Architectural Comparison

MLP and GRU policies achieved similar performance, indicating that the current execution state representation already captures most economically relevant information.

This suggests future gains are more likely to come from improved market-state features than from recurrent or transformer architectures.

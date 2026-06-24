# Stress-Test Interpretation

Stress tests use the existing simulator with config-level shocks. They are synthetic robustness diagnostics, not live-market performance claims.

## Tail-Risk Robustness

Across non-base stress scenarios, `Neural Policy` has the lowest average CVaR 95 at `200.29` bps.

## Neural Policy

Neural Policy has the lowest tail risk in liquidity_drought, volatility_spike, slow_transient_decay, late_day_liquidity_collapse, impact_misspecification.

## Classical Baselines

Among classical policies, the lowest-CVaR policy by stress scenario is: impact_misspecification: Recalibrated AC, late_day_liquidity_collapse: VWAP, liquidity_drought: Recalibrated AC, slow_transient_decay: Recalibrated AC, volatility_spike: Recalibrated AC.

## Liquidity Stress And Completion

No policy failed to complete the parent order in these scenarios. Participation-limit pressure appears in: liquidity_drought/VWAP, liquidity_drought/Almgren-Chriss, liquidity_drought/Neural Policy, late_day_liquidity_collapse/TWAP, late_day_liquidity_collapse/Almgren-Chriss, late_day_liquidity_collapse/Recalibrated AC.

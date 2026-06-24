# Stress-Test Interpretation

Stress tests use the existing simulator with config-level shocks. They are synthetic robustness diagnostics, not live-market performance claims.

## Tail-Risk Robustness

Across non-base stress scenarios, `VWAP` has the lowest average CVaR 95 at `630.60` bps.

## Neural Policy

Neural Policy has the lowest tail risk in volatility_spike, slow_transient_decay, impact_misspecification. Classical policies have lower CVaR in liquidity_drought, late_day_liquidity_collapse.

## Classical Baselines

Among classical policies, the lowest-CVaR policy by stress scenario is: impact_misspecification: Recalibrated AC, late_day_liquidity_collapse: VWAP, liquidity_drought: VWAP, slow_transient_decay: Recalibrated AC, volatility_spike: Recalibrated AC.

## Liquidity Stress And Completion

Completion failures appear in: base/TWAP, base/VWAP, base/Almgren-Chriss, base/Neural Policy, liquidity_drought/TWAP, liquidity_drought/VWAP, liquidity_drought/Almgren-Chriss, liquidity_drought/Recalibrated AC, liquidity_drought/Neural Policy, volatility_spike/TWAP, volatility_spike/VWAP, volatility_spike/Almgren-Chriss, volatility_spike/Neural Policy, slow_transient_decay/TWAP, slow_transient_decay/VWAP, slow_transient_decay/Almgren-Chriss, slow_transient_decay/Neural Policy, late_day_liquidity_collapse/TWAP, late_day_liquidity_collapse/VWAP, late_day_liquidity_collapse/Almgren-Chriss, late_day_liquidity_collapse/Recalibrated AC, late_day_liquidity_collapse/Neural Policy, impact_misspecification/TWAP, impact_misspecification/VWAP, impact_misspecification/Almgren-Chriss, impact_misspecification/Neural Policy.

# Neural Policy Heatmap Interpretation

The heatmaps show the saved neural policy's raw sigmoid output: the predicted fraction of the current maximum allowable trade. Actual shares are this fraction multiplied by `min(remaining_inventory, participation_rate * current_volume)`.

## Inventory And Time

- High inventory, short time remaining: `0.483`
- Low inventory, short time remaining: `0.479`
- High inventory, long time remaining: `0.485`

At short time remaining, the policy is nearly flat across low and high inventory. For high inventory, the policy is nearly flat across time remaining.

## Liquidity And Volatility

- High liquidity, neutral volatility: `0.478`
- Low liquidity, neutral volatility: `0.484`
- High volatility, neutral liquidity: `0.481`
- Low volatility, neutral liquidity: `0.483`

The policy fraction is nearly flat across the tested liquidity range; in share terms, higher liquidity still increases the max allowable trade. The policy is nearly flat across the tested volatility range.

## Transient Impact

- High transient impact: `0.482`
- Zero transient impact: `0.482`

The policy is nearly flat across the tested transient-impact range.

## Caveat

These diagnostics vary two normalized state dimensions at a time and hold the rest fixed at neutral values. They are local slices of the policy, not a full causal explanation of execution performance.

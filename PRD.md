PRD: Deep Optimal Execution under Market Impact and Liquidity Risk

Project summary

Project title Optimal Execution under Transient Market Impact and Stochastic Liquidity

One-line description a research framework that learns how to optimally split a large buy/sell order over time while accounting for market impact, volatility risk, stochastic liquidity, and tail execution risk.

Admissions-facing pitch traders cannot execute large orders instantly because trading too aggressively moves the market, while trading too slowly exposes them to price risk. This project studies the optimal execution problem using classical stochastic-control models and modern neural policies, comparing TWAP, VWAP, Almgren-Chriss, and deep risk-aware execution strategies.

Why this project fits your MS Quant Finance / Financial Engineering goal broadens you beyond option pricing into stochastic control, market microstructure, transaction costs, risk management, and deep learning for financial decision-making. The classic academic anchor is Almgren-Chriss, which frames execution as a tradeoff between market-impact cost and volatility risk; modern work extends this with deep learning and reinforcement learning under frictions and stochastic liquidity. ()

Problem statement

A trader needs to liquidate or acquire a large position (X_0) over a fixed horizon (T).

The trader faces a tradeoff:

[\text{Trade fast} \Rightarrow \text{high market impact}]

[\text{Trade slowly} \Rightarrow \text{high price risk}]

Classical models assume relatively simple impact and liquidity dynamics. In reality, liquidity changes intraday, impact may be transient, and execution cost has tail risk. A good execution model should adapt to market conditions instead of following a fixed schedule.

The project asks:

Can a neural stochastic-control policy reduce implementation shortfall and tail execution losses compared with TWAP, VWAP, and Almgren-Chriss under stochastic liquidity and transient market impact?

Goals

Primary goals

Implement a clean optimal execution simulation environment.

Implement classical baselines:

TWAP

VWAP

Almgren-Chriss

Dynamically recalibrated Almgren-Chriss

Model:

temporary impact

permanent impact

transient impact

stochastic volatility

stochastic liquidity

Train a neural execution policy that minimizes:

implementation shortfall

inventory risk

completion penalty

CVaR / tail execution risk

Evaluate policies across normal and stressed liquidity regimes.

Academic goals

The final project should read like a mini research paper:

“I compare classical stochastic-control execution models with neural policies under realistic liquidity and market-impact assumptions.”

This is much stronger than saying:

“I trained an RL trading agent.”

Non-goals

This project should not be framed as:

a trading bot

an alpha-generation system

stock price prediction

portfolio optimization

a live execution system connected to a broker

a high-frequency strategy

The goal is not to decide what to trade.The goal is to decide how to execute a known large order.

Target audience

Primary audience

Admissions committees and faculty reviewers for MS Quant Finance / Financial Engineering programs.

They should see:

stochastic processes

optimization

numerical simulation

market microstructure

risk-aware ML

strong empirical validation

Secondary audience

Quant research / execution research interviewers.

They should be able to ask you about:

Almgren-Chriss

implementation shortfall

transient market impact

CVaR

dynamic programming

neural stochastic control

liquidity stress testing

Literature foundation

The project should explicitly cite three layers of literature.

Classical execution

Almgren-Chriss is the core baseline. It models optimal execution as a balance between volatility risk and transaction costs from temporary and permanent market impact, and it constructs an efficient frontier of execution strategies. ()

Transient impact / market microstructure

Transient impact models matter because the cost of a trade may decay over time rather than remain permanent. Work on limit-order-book execution and transient price impact studies how splitting orders can reduce liquidity costs and how price impact depends on market resilience. ()

Deep learning / RL execution

Deep Hedging provides a useful conceptual bridge because it frames trading under transaction costs, market impact, liquidity constraints, and risk limits using deep reinforcement learning and convex risk measures. ()

A recent execution-specific paper studies reinforcement learning for optimal execution when liquidity is time-varying and latent, using a Double Deep Q-learning approach inside an Almgren-Chriss-style framework. ()

Core product requirements

P0: Minimum viable research product

The MVP should be enough to showcase in applications.

P0.1 Simulation environment

Build an execution environment where the agent must liquidate (X_0) shares over (N) time steps.

State:

[s_t = (q_t, t, S_t, \sigma_t, V_t, \eta_t)]

where:

(q_t): remaining inventory

(t): current time step

(S_t): unaffected midprice

(\sigma_t): volatility state

(V_t): market volume / liquidity state

(\eta_t): temporary impact coefficient

Action:

[u_t = \text{shares sold at time } t]

Constraints:

[0 \leq u_t \leq q_t]

[u_t \leq \rho V_t]

where (\rho) is a max participation-rate constraint.

Execution price for a sell order:

[P_t^{exec} = S_t - \eta_t u_t - Y_t]

where (Y_t) is transient impact state.

Inventory update:

[q_{t+1} = q_t - u_t]

Completion constraint:

[q_T = 0]

or impose a large terminal liquidation penalty.

P0.2 Classical baselines

Implement:

TWAPEqual-sized trades over time.

VWAPTrades proportional to expected volume curve.

Almgren-Chriss static scheduleClosed-form or numerical solution under constant volatility and impact.

Recalibrated Almgren-ChrissRecompute the schedule as volatility and liquidity estimates change.

The Almgren-Chriss baseline is essential. Without it, the project will look like generic RL.

P0.3 Cost function

Use implementation shortfall:

[IS = X_0S_0 - \sum_{t=0}^{N-1} u_t P_t^{exec}]

for a sell order.

Add risk and penalty terms:

[J =\mathbb{E}[IS]+\lambda \text{Var}(IS)+\alpha \text{CVaR}_{95}(IS)+\beta q_T^2+\gamma \sum_t \max(0, u_t - \rho V_t)^2]

This lets you compare average cost, variance, and tail cost.

P0.4 Neural execution policy

Train a neural policy:

[u_t = \pi_\theta(s_t)]

The policy should output a valid trade size.

Recommended architecture for MVP:

Input:remaining inventorytime remainingcurrent liquiditycurrent volatilitytransient impact staterecent returnsexpected volume bucket

Network:2-4 layer MLP

Output:participation rate or fraction of remaining inventory

Avoid complicated RL at first. The cleanest version is direct policy optimization through a differentiable simulator. RL can be added later.

P0.5 Evaluation metrics

Evaluate each policy on:

mean implementation shortfall

standard deviation of execution cost

95% CVaR

99% worst-tail cost

completion rate

participation-rate violations

average inventory over time

performance under liquidity shocks

sensitivity to impact-parameter misspecification

Do not use Sharpe ratio as the main metric. This is an execution-cost problem, not an alpha strategy.

P1: Strong admissions version

Once the MVP works, add the parts that make it look research-grade.

P1.1 Transient market impact

Add a transient impact state:

[Y_{t+1} = \rho Y_t + \phi u_t + \epsilon_t]

where:

(Y_t) is temporary price distortion caused by previous trades

(\rho) controls decay speed

(\phi) controls impact strength

Execution price:

[P_t^{exec} = S_t - \eta_t u_t - Y_t]

This is important because it makes the problem dynamic. The agent must learn that trading now affects future execution costs.

P1.2 Stochastic liquidity regimes

Model liquidity as a latent or observed regime:

Regime 0: normal liquidityRegime 1: high liquidityRegime 2: stressed liquidity

The regime controls:

market volume

spread proxy

temporary impact coefficient

volatility

impact decay rate

Example:

Normal:high volume, low impact, moderate volatility

Stress:low volume, high impact, high volatility

Event:high volume, high volatility, unstable impact

The recent RL literature specifically motivates this because liquidity is dynamic and often latent in real execution problems. ()

P1.3 Risk-aware training

Train two neural policies:

Mean-variance policy

[\min_\theta \mathbb{E}[IS] + \lambda \text{Var}(IS)]

CVaR-aware policy

[\min_\theta \mathbb{E}[IS] + \alpha \text{CVaR}_{95}(IS)]

Then show that the CVaR-aware policy may accept slightly higher average cost to reduce tail execution losses.

That is a very strong Financial Engineering result.

P1.4 Explainability

Add policy heatmaps.

Examples:

Trade rate as a function of inventory and time remaining.

Trade rate as a function of liquidity and volatility.

Trade rate as a function of transient impact state.

A good interpretation would be:

The neural policy accelerates when remaining inventory is high and time is short, but slows down when temporary impact is elevated and liquidity is poor.

This makes the deep learning explainable.

P2: Advanced version

These are optional extensions. Do not start here.

P2.1 Real market calibration

Calibrate the simulator from real intraday data.

Preferred sources:

WRDS TAQ, which provides tick-by-tick trades and quotes for U.S. National Market System activity with intraday timestamps down to the microsecond. ()

Databento MBO, which provides order-book events keyed by order ID, including trades, fills, adds, cancels, modifies, and clears. ()

LOBSTER, which provides reconstructed Nasdaq limit-order-book data and sample files with message and orderbook files based on Nasdaq Historical TotalView-ITCH samples. ()

For admissions, WRDS TAQ is enough. Databento MBO or LOBSTER is stronger but more engineering-heavy.

P2.2 Neural liquidity filter

Make liquidity partially observed.

The model does not directly observe true (\eta_t) or regime (L_t). Instead, it observes noisy proxies:

volumespread proxyrecent volatilityrecent price impacttrade imbalance

Then add a small recurrent encoder:

[\hat{L}t = f\theta(\text{recent market observations})]

Policy:

[u_t = \pi_\theta(q_t, t, \hat{L}_t)]

This becomes a partial-observation stochastic-control problem.

P2.3 Robust optimization

Train under one set of parameters and test under shifted parameters.

Stress-test:

impact coefficient +50%volume -50%volatility +100%impact decay slower than expectedliquidity regime switches unexpectedly

This is great for admissions because it shows awareness of model risk.

Data strategy

Recommended data plan

Phase 1: Synthetic calibrated environment

Start with simulated paths.

Use stylized intraday profiles:

U-shaped volume curveU-shaped volatility curveregime-switching liquiditytemporary/permanent/transient impact

This lets you prove the control framework without spending weeks on data cleaning.

Phase 2: Empirical calibration

Estimate parameters from intraday data:

intraday volume curverealized volatility by time bucketspread proxyshort-term price impact proxyvolume-volatility relationship

WRDS TAQ is ideal if you can access it. If not, use a public or vendor intraday dataset and clearly state limitations.

Phase 3: Optional order-book data

Use LOBSTER or Databento only if you want queue/depth-aware execution. That is not necessary for the core admissions project.

Functional requirements

ID

Requirement

Priority

FR1

User can configure parent order size, horizon, number of intervals, side, risk aversion, and impact parameters.

P0

FR2

System simulates unaffected price paths with stochastic volatility.

P0

FR3

System simulates temporary and permanent impact.

P0

FR4

System supports TWAP, VWAP, and Almgren-Chriss baselines.

P0

FR5

System trains a neural policy to minimize execution cost.

P0

FR6

System reports implementation shortfall, variance, CVaR, and completion rate.

P0

FR7

System supports transient impact decay.

P1

FR8

System supports stochastic liquidity regimes.

P1

FR9

System supports stress testing and parameter misspecification.

P1

FR10

System generates publication-quality plots and tables.

P1

FR11

System calibrates volume/volatility/impact proxies from intraday data.

P2

FR12

System supports partially observed liquidity with recurrent filtering.

P2

Non-functional requirements

Requirement

Description

Reproducibility

All experiments should be config-driven with fixed random seeds.

Modularity

Separate simulator, policies, baselines, training, evaluation, and plotting.

Interpretability

Include policy heatmaps and stress-test explanations.

Speed

MVP should train on a laptop GPU or Colab within reasonable time.

Academic rigor

Include mathematical formulation, baselines, ablations, and limitations.

No leakage

Policies cannot observe future volume, volatility, or shocks unless explicitly modeled as forecast inputs.

No live trading

No broker integration or live order submission.

Model design

Environment

At each step:

Observe state (s_t).

Choose trade size (u_t).

Compute execution price.

Update cash and inventory.

Update price, volatility, liquidity, and transient impact.

Continue until (T).

Baseline policies

TWAP:u_t = q_t / remaining_steps

VWAP:u_t proportional to expected volume curve

Almgren-Chriss:closed-form or numerical optimal schedule under constant parameters

Recalibrated AC:AC schedule recomputed when volatility or liquidity changes

Neural policy

Recommended first version:

MLP policy:input_dim = 8-15hidden_layers = [128, 128, 64]activation = ReLU or SiLUoutput = fraction of max allowable trade

Output transformation:

[u_t = \min(q_t, \rho V_t) \cdot \sigma(a_t)]

where (\sigma(a_t)) is sigmoid output.

Training objective

Use Monte Carlo simulation:

[\min_\theta\frac{1}{M}\sum_{m=1}^M IS_m+\lambda \widehat{\text{Var}}(IS)+\alpha \widehat{\text{CVaR}}_{95}(IS)+\beta \mathbb{E}[q_T^2]]

Evaluation design

Main experiments

Experiment 1: Classical sanity check

Compare TWAP, VWAP, and Almgren-Chriss under constant liquidity.

Expected result:

Almgren-Chriss should produce a sensible cost-risk frontier.Higher risk aversion should trade faster.

Experiment 2: Stochastic liquidity

Compare static AC vs recalibrated AC vs neural policy.

Expected result:

Neural policy should adapt better when liquidity changes over time.

Experiment 3: Transient impact

Test whether the neural policy learns to avoid over-trading when previous trades have created large temporary price pressure.

Expected result:

Policy should smooth trades when transient impact is high.

Experiment 4: Tail-risk objective

Compare mean-variance neural policy vs CVaR-aware neural policy.

Expected result:

CVaR policy may have slightly higher mean cost but lower worst-case cost.

Experiment 5: Stress testing

Evaluate all policies under:

liquidity droughtvolatility spikeimpact coefficient misspecificationlate-day volume collapseregime switch

Expected result:

Risk-aware neural policy should be more robust in stress regimes.

Success metrics

The project is successful if it produces these outputs:

Technical success

Almgren-Chriss baseline behaves correctly.

Neural policy outperforms TWAP/VWAP in stochastic liquidity settings.

CVaR-aware policy reduces tail execution costs.

Stress-test results are interpretable.

Learned policy heatmaps make economic sense.

Admissions success

The final report clearly demonstrates:

stochastic control

numerical simulation

market impact modeling

risk-aware optimization

neural policy learning

rigorous baseline comparison

Portfolio success

A reviewer should be able to understand the project in one minute:

“He built a framework for optimal execution, compared classical financial engineering models with neural stochastic-control policies, and evaluated execution cost and tail risk under liquidity stress.”

Deliverables

Required deliverables

GitHub repository

Research report PDF

Clean README

Experiment configs

Plots and result tables

Reproducible notebook or script

Recommended repo structure

deep-optimal-execution/README.mdreport/optimal_execution_report.pdfreferences.bibconfigs/ac_baseline.yamlstochastic_liquidity.yamlcvar_policy.yamlsrc/environment/execution_env.pyprice_process.pyimpact_models.pyliquidity_models.pypolicies/twap.pyvwap.pyalmgren_chriss.pyneural_policy.pytraining/train_policy.pylosses.pyevaluation/metrics.pystress_tests.pyplots.pynotebooks/01_ac_sanity_check.ipynb02_neural_policy_training.ipynb03_stress_testing.ipynbresults/figures/tables/

Final report structure

Use a mini-dissertation format.

Abstract

Introduction

Literature Review

Mathematical Formulation

Market Impact and Liquidity Model

Classical Baselines

Neural Policy Methodology

Experimental Design

Results

Stress Testing

Limitations

Conclusion

The abstract could be:

This project studies the optimal execution problem under market impact and stochastic liquidity. I implement classical TWAP, VWAP, and Almgren-Chriss baselines, extend the environment to transient impact and regime-switching liquidity, and train neural policies to minimize implementation shortfall under mean-variance and CVaR objectives. Experiments show how adaptive policies respond to liquidity shocks, volatility changes, and impact misspecification, highlighting the tradeoff between average execution cost and tail execution risk.

Timeline

Week 1: Literature and mathematical setup

Read Almgren-Chriss.

Define execution cost, impact model, inventory dynamics.

Write initial project specification.

Week 2: Simulator MVP

Build price process.

Build inventory and cash accounting.

Add temporary and permanent impact.

Validate TWAP behavior.

Week 3: Classical baselines

Implement VWAP.

Implement Almgren-Chriss.

Plot efficient frontier.

Verify risk-aversion behavior.

Week 4: Stochastic liquidity

Add volume curve.

Add liquidity regimes.

Add volatility regimes.

Test baselines under regime changes.

Week 5: Neural policy

Implement MLP policy.

Train with mean-variance objective.

Compare against baselines.

Week 6: Transient impact

Add impact decay state.

Retrain neural policy.

Add policy heatmaps.

Week 7: CVaR training

Implement tail-risk objective.

Compare mean-variance vs CVaR policy.

Analyze execution-cost distributions.

Week 8: Stress tests

Liquidity drought.

Volatility spike.

Impact misspecification.

Regime-switch stress.

Week 9: Empirical calibration

Calibrate volume and volatility curves from available intraday data.

Replace purely synthetic profiles with empirically grounded profiles.

Week 10: Report and polish

Write final PDF.

Clean GitHub.

Add resume bullets.

Create final figures.

Risks and mitigations

Risk

Why it matters

Mitigation

Neural policy fails to beat AC

Classical models are strong in simple settings

Use stochastic liquidity and transient impact where static AC is less suitable

Project looks like generic RL

Admissions reviewers may dislike trading-bot framing

Emphasize stochastic control, execution cost, and risk metrics

Simulator seems unrealistic

Synthetic data can look toy-like

Calibrate volume/volatility profiles from intraday data

Too much engineering

LOB data can consume time

Start with calibrated simulation; add TAQ/MBO only later

Results are unstable

RL can be noisy

Start with differentiable policy optimization before full RL

Too many extensions

Scope creep

Ship P0 + P1 before touching P2

Resume entry

Use this as the final resume project:

Deep Optimal Execution under Market Impact and Liquidity Risk

Built a stochastic optimal-execution framework comparing TWAP, VWAP, Almgren-Chriss, dynamic recalibration, and neural control policies under temporary, permanent, and transient market impact.

Modeled stochastic volatility, intraday volume, liquidity regimes, and execution constraints; trained risk-aware neural policies to minimize implementation shortfall, variance, CVaR, and completion penalties.

Evaluated policies through efficient frontiers, execution-cost distributions, stress tests, and learned policy heatmaps under liquidity shocks and impact-parameter misspecification.

The cleanest MVP

The first version should be:

Simulated execution environment

TWAP

VWAP

Almgren-Chriss

stochastic liquidity

neural policy

implementation shortfall / CVaR evaluation

stress-test plots

Do not start with order-book reconstruction. Do not start with live data. Do not start with complicated RL.

Start with a mathematically clean simulator and strong baselines. Then add complexity.

This project will broaden your profile in exactly the right direction: from option pricing into stochastic control, market impact, liquidity risk, and execution research.
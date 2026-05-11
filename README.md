# Derivative Hedging via Deep Reinforcement Learning

> A DDPG-based autonomous agent that learns dynamic option hedging strategies — benchmarked against the classical Black-Scholes delta hedge.

---

## Overview

Classical options hedging relies on the **Black-Scholes delta** — a closed-form formula that tells you exactly how much of the underlying stock to hold at each moment. It is elegant and mathematically rigorous, but it assumes constant volatility, frictionless markets, and continuous rebalancing. None of these hold in practice.

This project replaces the formula with a **learned policy**. A Deep Deterministic Policy Gradient (DDPG) agent observes live market state and decides the hedge ratio every trading day. It receives a reward signal that directly encodes the hedging objective: *make your hedge gains cancel your option liability, as cheaply as possible.* No formula is given — the agent discovers the strategy from experience.

The agent is trained on simulated Geometric Brownian Motion price paths and evaluated head-to-head against the Black-Scholes benchmark on identical paths, making the comparison fair.

---

## Motivation

| Assumption | Black-Scholes Requires | Reality |
|---|---|---|
| Volatility | Constant | Stochastic, mean-reverting |
| Trading | Continuous | Discrete, costly |
| Transaction costs | Zero | 0.1%+ per rebalance |
| Market impact | None | Real and path-dependent |

A reinforcement learning agent trained under realistic conditions can — in principle — learn to **adapt its hedging frequency and aggressiveness** based on moneyness, time-to-expiry, and transaction cost trade-offs. This is something a static formula cannot do.

---

## The Setup

- **Underlying:** Infosys stock, S₀ = ₹1500
- **Option:** Short European Call, Strike K = ₹1550 (slightly OTM), 3-month maturity
- **Trading:** 90 discrete daily steps over the option's life
- **Transaction cost:** 0.1% of trade value per rebalance
- **Volatility:** σ = 25% (implied), Risk-free rate r = 5%

---

## Architecture

### System Overview

```
config.py  ──────────────────────────────────────────────────────────┐
                                                                      │
MarketSimulator                                                        │
  └── simulate_gbm()          Generates 90-day GBM stock path        │
         │                                                            │
         ▼                                                            │
OptionHedgingEnv               Gym-style environment                 ▼
  ├── reset()                  New path, reset position         DDPGAgent
  ├── step(action)             Compute reward, advance time       ├── Actor Network
  │     reward = stock_pnl                                        │     4 → 128 → 128 → 64 → 1
  │           - option_pnl                                        │     (Sigmoid output ∈ [0,1])
  │           - trade_cost                                        │
  └── run_bs_benchmark()       Replay path with BS delta         ├── Critic Network
                                                                  │     5 → 128 → 128 → 64 → 1
                                                                  │     (Raw Q-value)
black_scholes.py                                                  │
  ├── bs_call_price()  ────────────────────────────────────────── ┤
  └── bs_delta()       ← benchmark action                        └── ReplayBuffer + OUNoise
```

---

### State Space (4-dimensional)

| Dimension | Formula | What it captures |
|---|---|---|
| Normalized price | `S / S₀` | Where the stock is relative to start |
| Normalized TTM | `ttm / T` | How much of the option's life remains |
| Current delta | `current_delta` | What the agent already holds |
| Log-moneyness | `log(S / K)` | How deep in/out of the money the option is |

### Action Space

| | Value |
|---|---|
| Type | Continuous |
| Range | [0, 1] |
| Meaning | Fraction of one share to hold as hedge |

### Reward Function

```
R_t = (new_delta × ΔS)  −  (V_{t+1} − V_t)  −  (0.001 × |Δhedge| × S_t)
       ▲                     ▲                    ▲
  Gain from hedge       Option liability     Transaction cost
  position              change (liability)   of rebalancing
```

Maximizing cumulative reward is equivalent to minimizing total hedging cost over the option's life.

---

### DDPG Algorithm

DDPG (Deep Deterministic Policy Gradient) is an actor-critic algorithm designed for **continuous action spaces**. It maintains four networks:

| Network | Role |
|---|---|
| Actor `π(s)` | Outputs hedge ratio given current state |
| Critic `Q(s, a)` | Estimates expected future reward for (state, action) pair |
| Actor Target `π'(s)` | Slowly-tracked copy of actor — stabilizes training |
| Critic Target `Q'(s, a)` | Slowly-tracked copy of critic — stabilizes Bellman targets |

**Update cycle (every step):**

1. Critic update: minimize MSE between `Q(s,a)` and TD-target `r + γ · Q'(s', π'(s'))`
2. Actor update: ascend the Q-gradient `∇_a Q(s, π(s))`
3. Soft target update: `θ' ← τ·θ + (1−τ)·θ'` with τ = 0.005

**Exploration:** Ornstein-Uhlenbeck noise added to actor output during training. OU noise is temporally correlated — it drifts smoothly — encouraging the agent to explore nearby hedge ratios rather than jumping wildly.

---

## Project Structure

```
derivative_hedging/
│
├── config.py                  # All hyperparameters — single source of truth
├── train.py                   # Training loop (2000 episodes)
├── evaluate.py                # RL vs Black-Scholes benchmark
├── requirements.txt
│
├── agent/
│   ├── ddpg_agent.py          # Core DDPG logic: select, update, save, load
│   ├── networks.py            # Actor and Critic neural network definitions
│   ├── replay_buffer.py       # Circular experience replay buffer (200K cap)
│   └── noise.py               # Ornstein-Uhlenbeck exploration noise
│
├── environment/
│   ├── option_env.py          # Gym-style hedging environment
│   └── market_simulator.py    # GBM stock path generator
│
├── utils/
│   └── black_scholes.py       # bs_call_price() and bs_delta() — benchmark
│
└── output/                    # Generated at runtime
    ├── ddpg_hedging.pth       # Trained model checkpoint
    ├── training_log.csv       # Per-episode reward and loss
    └── evaluation_results.csv # RL vs BS cost per episode
```

---

## Hyperparameters

### Market Parameters

| Parameter | Value | Description |
|---|---|---|
| S₀ | ₹1500 | Initial stock price |
| K | ₹1550 | Strike price (OTM) |
| σ | 25% | Implied volatility |
| r | 5% | Risk-free rate (annual) |
| T | 0.25 years | Time to maturity (3 months) |
| dt | 1/90 | One trading day |
| Transaction cost | 0.1% | Per rebalance, on trade value |

### DDPG Hyperparameters

| Parameter | Value | Description |
|---|---|---|
| State dim | 4 | Input to both networks |
| Action dim | 1 | Hedge ratio ∈ [0, 1] |
| Actor LR | 1e-4 | Adam optimizer |
| Critic LR | 3e-4 | Adam optimizer |
| γ (discount) | 0.99 | Future reward weighting |
| τ (soft update) | 0.005 | Target network tracking rate |
| Batch size | 256 | Transitions per update |
| Buffer capacity | 200,000 | Max stored transitions |
| Warmup episodes | 20 | Random actions before training begins |
| Gradient clip | 1.0 (norm) | Applied to both actor and critic |

---

## Installation

```bash
git clone https://github.com/your-username/derivative-hedging.git
cd derivative-hedging

python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS/Linux

pip install -r requirements.txt
```

---

## Usage

### Train

```bash
python train.py
```

Trains the DDPG agent for 2000 episodes. Prints a checkpoint every 100 episodes and saves the model to `output/ddpg_hedging.pth`.

```
Training for 2000 episodes...
 Episode |  EP Reward |  Critic Loss |  Actor Loss
----------------------------------------------------
     100 |     100.42 |      176.598 |     -14.336
     200 |      56.00 |      249.449 |     -28.274
     ...
    2000 |     -55.14 |      233.919 |     -12.887
Model saved -> output/ddpg_hedging.pth
Training complete. Log saved -> output/training_log.csv
```

### Evaluate

```bash
python evaluate.py
```

Runs 50 episodes with the trained agent and the BS benchmark on identical stock paths.

```
-------------------------------------------------
  RL Agent -> Cost: Rs.0.74 +/- Rs.62.82
  BS Hedge -> Cost: Rs.-6.69 +/- Rs.14.64
  Improvement: 111.1% lower cost with RL
-------------------------------------------------
Results saved -> output/evaluation_results.csv
```

---

## Results

### Training Progression (2000 Episodes)

| Episode | Reward | Critic Loss | Actor Loss |
|---|---|---|---|
| 100 | +100.42 | 176.6 | -14.3 |
| 300 | +61.01 | 201.9 | -22.7 |
| 500 | -33.92 | 229.0 | -23.3 |
| 700 | -80.32 | 256.6 | -18.8 |
| 1000 | +94.67 | 176.4 | -16.1 |
| 1300 | +97.05 | 196.2 | -12.1 |
| 1600 | +103.34 | 216.5 | -13.2 |
| 2000 | -55.14 | 233.9 | -12.9 |

### Evaluation: RL vs Black-Scholes (50 Episodes)

| Strategy | Mean Hedging Cost | Std Dev |
|---|---|---|
| DDPG Agent | Rs. 0.74 | ± Rs. 62.82 |
| Black-Scholes Delta | Rs. -6.69 | ± Rs. 14.64 |

### Analysis

The critic loss failing to converge (staying 176–257 throughout all 2000 episodes) is the clearest signal that the Q-function never stabilized. Without a reliable critic, the actor receives poor gradient signals and exploits Q-value overestimates rather than learning true hedging behaviour — a known failure mode of vanilla DDPG called **overestimation bias**.

The RL agent's near-zero mean cost looks encouraging at first glance, but its standard deviation (±62.82) is **4.3× higher** than Black-Scholes (±14.64), revealing that the agent is inconsistent and path-sensitive rather than robustly hedged.

Black-Scholes achieves a slightly negative mean cost — a mild profit from the hedge — with tight variance, which is the hallmark of a well-calibrated strategy.

---

## Limitations and Next Steps

| Issue | Root Cause | Proposed Fix |
|---|---|---|
| Critic doesn't converge | Noisy stochastic reward; Q-overestimation | Switch to **TD3** (Twin Delayed DDPG) — clips Q-targets to remove overestimation |
| High agent variance | Reward conflates market noise with hedge quality | **Reward normalization** or use PnL *residual* from BS delta as reward |
| Too few episodes | 2000 × 90 steps = 180K transitions (thin for continuous control) | Train for 5k–10k episodes |
| Agent may learn degenerate policy | Positive drift in GBM rewards always-long positions | Add **stochastic volatility** (Heston model) or regime-switching paths |
| No risk penalty | Agent ignores variance of cost, only mean | Add **CVaR or variance penalty** to reward function |

---

## Concepts Covered

- Deep Reinforcement Learning (Actor-Critic, DDPG)
- Options pricing and the Greeks (delta hedging)
- Geometric Brownian Motion simulation
- Black-Scholes model and its assumptions
- Experience replay and target networks
- Ornstein-Uhlenbeck exploration noise
- Transaction cost modelling in RL reward design

---

## References

- Lillicrap et al. (2016) — *Continuous Control with Deep Reinforcement Learning* (DDPG paper)
- Black & Scholes (1973) — *The Pricing of Options and Corporate Liabilities*
- Merton (1973) — *Theory of Rational Option Pricing*
- Kolm & Ritter (2019) — *Dynamic Replication and Hedging: A Reinforcement Learning Approach*

---

## License

MIT License. See `LICENSE` for details.

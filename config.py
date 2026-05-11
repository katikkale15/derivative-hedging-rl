# ── All hyperparameters in one place ──────────────────────────────────────────

# Market / Option Parameters
S0              = 1500      # Initial stock price (Infosys)
K               = 1550      # Strike price
r               = 0.05      # Risk-free rate (5% annually)
sigma           = 0.25      # Implied volatility (25%)
T               = 0.25      # Time to maturity (3 months = 0.25 years)
dt              = 1 / 90    # One trading day step
TRANSACTION_COST = 0.001    # 0.1% of trade value per rebalance

# DDPG Agent Hyperparameters
STATE_DIM       = 4         # [norm_price, ttm, current_delta, moneyness]
ACTION_DIM      = 1         # hedge ratio ∈ [0, 1]
ACTOR_LR        = 1e-4      # Learning rate for Actor network
CRITIC_LR       = 3e-4      # Learning rate for Critic network
GAMMA           = 0.99      # Discount factor for future rewards
TAU             = 0.005     # Soft update rate for target networks
BATCH_SIZE      = 256       # Transitions sampled per training update
BUFFER_CAPACITY = 200_000   # Max transitions stored in replay buffer

# Training Loop
EPISODES        = 2000      # Total training episodes
WARMUP_EPISODES = 20        # Episodes before training begins (fill buffer first)
EVAL_EVERY      = 100       # Evaluate every N episodes
EVAL_EPISODES   = 50        # Episodes used for evaluation

# Paths
MODEL_SAVE_PATH = "output/ddpg_hedging.pth"
LOG_SAVE_PATH   = "output/training_log.csv"
import numpy as np
from environment.market_simulator import simulate_gbm
from utils.black_scholes import bs_call_price, bs_delta
import config

class OptionHedgingEnv:
    """
    Custom RL environment for hedging a short call option.

    State  (4,): [normalized_price, normalized_ttm, current_delta, moneyness]
    Action (1,): new hedge ratio ∈ [0, 1]
    Reward     : -(hedge_pnl_error + transaction_cost) per step
    """

    def __init__(self):
        self.S0      = config.S0
        self.K       = config.K
        self.r       = config.r
        self.sigma   = config.sigma
        self.T       = config.T
        self.dt      = config.dt
        self.tc      = config.TRANSACTION_COST
        self.n_steps = int(config.T / config.dt)

    def reset(self):
        self.stock_path    = simulate_gbm(self.S0, self.r, self.sigma, self.T, self.dt)
        self.t             = 0
        self.current_delta = 0.0
        return self._get_state()

    def _get_state(self):
        S   = self.stock_path[self.t]
        ttm = (self.n_steps - self.t) * self.dt
        return np.array([
            S / self.S0,                    # normalized price
            ttm / self.T,                   # normalized time to expiry
            self.current_delta,             # current hedge position
            np.log(S / self.K)             # log-moneyness
        ], dtype=np.float32)

    def step(self, action):
        new_delta = float(np.clip(action, 0.0, 1.0))

        S_curr   = self.stock_path[self.t]
        S_next   = self.stock_path[self.t + 1]
        ttm_curr = (self.n_steps - self.t)     * self.dt
        ttm_next = (self.n_steps - self.t - 1) * self.dt

        # Gain from holding hedge position
        stock_pnl  = new_delta * (S_next - S_curr)

        # Change in option value (our liability as the seller)
        V_curr     = bs_call_price(S_curr, self.K, self.r, self.sigma, ttm_curr)
        V_next     = bs_call_price(S_next, self.K, self.r, self.sigma, ttm_next)
        option_pnl = V_next - V_curr

        # Brokerage cost for changing hedge
        trade_cost = self.tc * abs(new_delta - self.current_delta) * S_curr

        # Reward: hedge gain should cancel option liability, minus costs
        reward = stock_pnl - option_pnl - trade_cost

        self.current_delta = new_delta
        self.t += 1
        done = (self.t >= self.n_steps)
        return self._get_state(), reward, done

    def run_bs_benchmark(self):
        """Run one full episode using Black-Scholes delta on the same stock path as the RL episode."""
        # Reuse existing stock_path — only reset position counters
        self.t = 0
        self.current_delta = 0.0
        total_cost = 0.0
        done = False
        while not done:
            S   = self.stock_path[self.t]
            ttm = (self.n_steps - self.t) * self.dt
            action = bs_delta(S, self.K, self.r, self.sigma, ttm)
            _, reward, done = self.step(action)
            total_cost -= reward
        return total_cost
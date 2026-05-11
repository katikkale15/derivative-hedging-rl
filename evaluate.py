import os
import numpy as np
import pandas as pd
import config
from environment.option_env import OptionHedgingEnv
from agent.ddpg_agent import DDPGAgent

def evaluate():
    os.makedirs("output", exist_ok=True)
    env   = OptionHedgingEnv()
    agent = DDPGAgent()
    agent.load()

    rl_costs, bs_costs = [], []

    print(f"Running {config.EVAL_EPISODES} evaluation episodes...")
    for ep in range(config.EVAL_EPISODES):
        # ── RL Agent Episode ──────────────────────────────────────────────
        state = env.reset()
        done  = False
        rl_cost = 0.0
        while not done:
            action = agent.select_action(state, explore=False)
            state, reward, done = env.step(action)
            rl_cost -= reward
        rl_costs.append(rl_cost)

        # ── BS Benchmark Episode (same stock path) ────────────────────────
        bs_cost = env.run_bs_benchmark()
        bs_costs.append(bs_cost)

    rl_mean, rl_std = np.mean(rl_costs), np.std(rl_costs)
    bs_mean, bs_std = np.mean(bs_costs), np.std(bs_costs)
    improvement     = (bs_mean - rl_mean) / bs_mean * 100

    print("\n" + "-" * 49)
    print(f"  RL Agent -> Cost: Rs.{rl_mean:.2f} +/- Rs.{rl_std:.2f}")
    print(f"  BS Hedge -> Cost: Rs.{bs_mean:.2f} +/- Rs.{bs_std:.2f}")
    print(f"  Improvement: {improvement:.1f}% lower cost with RL")
    print("-" * 49)

    pd.DataFrame({
        'episode': range(1, config.EVAL_EPISODES + 1),
        'rl_cost': rl_costs,
        'bs_cost': bs_costs
    }).to_csv("output/evaluation_results.csv", index=False)
    print("Results saved -> output/evaluation_results.csv")

if __name__ == "__main__":
    evaluate()
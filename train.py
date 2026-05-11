import numpy as np
import pandas as pd
import os
import config
from environment.option_env import OptionHedgingEnv
from agent.ddpg_agent import DDPGAgent

def train():
    os.makedirs("output", exist_ok=True)
    env   = OptionHedgingEnv()
    agent = DDPGAgent()
    log   = []

    print(f"Training for {config.EPISODES} episodes...")
    print(f"{'Episode':>8} | {'EP Reward':>10} | {'Critic Loss':>12} | {'Actor Loss':>11}")
    print("-" * 52)

    for ep in range(1, config.EPISODES + 1):

        state = env.reset()
        agent.noise.reset()
        ep_reward   = 0.0
        critic_loss = None
        actor_loss  = None

        done = False
        while not done:
            if ep <= config.WARMUP_EPISODES:
                action = float(np.random.uniform(0.0, 1.0))
            else:
                action = agent.select_action(state, explore=True)

            next_state, reward, done = env.step(action)
            agent.buffer.push(state, action, reward, next_state, done)
            state      = next_state
            ep_reward += reward

            c_loss, a_loss = agent.update()
            if c_loss is not None:
                critic_loss = c_loss
                actor_loss  = a_loss

        log.append({
            'episode':     ep,
            'reward':      round(ep_reward, 4),
            'critic_loss': round(critic_loss, 6) if critic_loss is not None else None,
            'actor_loss':  round(actor_loss,  6) if actor_loss  is not None else None,
        })

        if ep % config.EVAL_EVERY == 0:
            print(f"{ep:>8} | {ep_reward:>10.2f} | "
                  f"{critic_loss if critic_loss is not None else 'N/A':>12} | "
                  f"{actor_loss  if actor_loss  is not None else 'N/A':>11}")
            agent.save()

    # Save training log
    pd.DataFrame(log).to_csv(config.LOG_SAVE_PATH, index=False)
    print(f"\nTraining complete. Log saved -> {config.LOG_SAVE_PATH}")

if __name__ == "__main__":
    train()
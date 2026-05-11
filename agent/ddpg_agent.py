import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os

from agent.networks import Actor, Critic
from agent.noise import OUNoise
from agent.replay_buffer import ReplayBuffer
import config

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DDPGAgent:
    def __init__(self):
        # Live networks
        self.actor  = Actor(config.STATE_DIM).to(device)
        self.critic = Critic(config.STATE_DIM, config.ACTION_DIM).to(device)

        # Target networks (slowly track live networks)
        self.actor_target  = Actor(config.STATE_DIM).to(device)
        self.critic_target = Critic(config.STATE_DIM, config.ACTION_DIM).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_opt  = optim.Adam(self.actor.parameters(),  lr=config.ACTOR_LR)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=config.CRITIC_LR)

        self.buffer = ReplayBuffer()
        self.noise  = OUNoise()

    def select_action(self, state, explore=True):
        s = torch.FloatTensor(state).unsqueeze(0).to(device)
        self.actor.eval()
        with torch.no_grad():
            action = self.actor(s).cpu().item()
        self.actor.train()
        if explore:
            action += self.noise.sample() * 0.1
        return float(np.clip(action, 0.0, 1.0))

    def update(self):
        if len(self.buffer) < config.BATCH_SIZE:
            return None, None

        s, a, r, ns, d = self.buffer.sample()

        S  = torch.FloatTensor(s).to(device)
        A  = torch.FloatTensor(a).unsqueeze(1).to(device)
        R  = torch.FloatTensor(r).unsqueeze(1).to(device)
        NS = torch.FloatTensor(ns).to(device)
        D  = torch.FloatTensor(d).unsqueeze(1).to(device)

        # ── Critic Update ──────────────────────────────────────────────────
        with torch.no_grad():
            next_actions = self.actor_target(NS)
            target_q     = R + config.GAMMA * (1 - D) * self.critic_target(NS, next_actions)

        current_q   = self.critic(S, A)
        critic_loss = nn.MSELoss()(current_q, target_q)

        self.critic_opt.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_opt.step()

        # ── Actor Update ───────────────────────────────────────────────────
        actor_loss = -self.critic(S, self.actor(S)).mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_opt.step()

        # ── Soft Update of Target Networks ─────────────────────────────────
        for p, tp in zip(self.actor.parameters(), self.actor_target.parameters()):
            tp.data.copy_(config.TAU * p.data + (1 - config.TAU) * tp.data)
        for p, tp in zip(self.critic.parameters(), self.critic_target.parameters()):
            tp.data.copy_(config.TAU * p.data + (1 - config.TAU) * tp.data)

        return critic_loss.item(), actor_loss.item()

    def save(self):
        os.makedirs("output", exist_ok=True)
        torch.save({
            'actor':         self.actor.state_dict(),
            'critic':        self.critic.state_dict(),
            'actor_target':  self.actor_target.state_dict(),
            'critic_target': self.critic_target.state_dict(),
        }, config.MODEL_SAVE_PATH)
        print(f"Model saved -> {config.MODEL_SAVE_PATH}")

    def load(self):
        checkpoint = torch.load(config.MODEL_SAVE_PATH, map_location=device, weights_only=True)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.actor_target.load_state_dict(checkpoint['actor_target'])
        self.critic_target.load_state_dict(checkpoint['critic_target'])
        print(f"Model loaded <- {config.MODEL_SAVE_PATH}")
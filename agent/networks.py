import torch
import torch.nn as nn

class Actor(nn.Module):
    """
    Policy network — maps state → hedge ratio.
    Input  : 4-dim state vector
    Output : single value in [0, 1] via Sigmoid (the hedge ratio)
    """
    def __init__(self, state_dim=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()                    # constrains output to [0, 1]
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, state):
        return self.net(state)


class Critic(nn.Module):
    """
    Q-value network — maps (state, action) → expected total future reward.
    Input  : state (4) concatenated with action (1) = 5-dim input
    Output : single scalar Q-value (no activation — can be negative)
    """
    def __init__(self, state_dim=4, action_dim=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)                # raw Q-value
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        return self.net(x)
import numpy as np

class OUNoise:
    """
    Ornstein-Uhlenbeck process noise for smooth exploration.
    Unlike pure random noise, OU noise is temporally correlated —
    it drifts gradually, encouraging the agent to explore nearby
    hedge ratios rather than jumping wildly.
    """
    def __init__(self, mu=0.0, theta=0.15, sigma=0.2):
        self.mu    = mu
        self.theta = theta
        self.sigma = sigma
        self.reset()

    def reset(self):
        self.state = self.mu

    def sample(self):
        dx = self.theta * (self.mu - self.state) + self.sigma * np.random.randn()
        self.state += dx
        return self.state
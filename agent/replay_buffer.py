import numpy as np
import random
from collections import deque
import config

class ReplayBuffer:
    """
    Circular memory bank storing past agent experiences.
    Random sampling breaks temporal correlations, which
    prevents the neural network from overfitting to recent episodes.
    """
    def __init__(self):
        self.buffer = deque(maxlen=config.BUFFER_CAPACITY)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((
            state, float(action), float(reward), next_state, float(done)
        ))

    def sample(self, batch_size=None):
        bs    = batch_size or config.BATCH_SIZE
        batch = random.sample(self.buffer, bs)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states,      dtype=np.float32),
            np.array(actions,     dtype=np.float32),
            np.array(rewards,     dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones,       dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)
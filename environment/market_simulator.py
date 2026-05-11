import numpy as np

def simulate_gbm(S0, mu, sigma, T, dt):
    """
    Simulate a single stock price path using Geometric Brownian Motion.
    Returns array of shape (n_steps + 1,) — includes starting price.
    """
    n_steps = int(T / dt)
    Z = np.random.standard_normal(size=n_steps)
    S = np.zeros(n_steps + 1)
    S[0] = S0
    for t in range(n_steps):
        S[t + 1] = S[t] * np.exp(
            (mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z[t]
        )
    return S
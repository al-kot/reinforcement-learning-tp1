from __future__ import annotations

from typing import Iterable, Sequence
import numpy as np


def cumulative_regret(means: Sequence[float], rewards: Iterable[int]) -> np.ndarray:
    """Compute cumulative regret R_t = t * mu* - cumulative_reward(t)."""
    mu_star = float(max(means))
    rewards_arr = np.asarray(list(rewards), dtype=float)
    cum_reward = rewards_arr.cumsum()
    t = np.arange(1, len(rewards_arr) + 1)
    optimal = mu_star * t
    return optimal - cum_reward

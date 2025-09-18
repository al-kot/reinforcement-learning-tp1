from __future__ import annotations

from typing import List, Tuple
import numpy as np

from .arms import ArmBernoulli


def naive(
    mab: List[ArmBernoulli], T: int = 100, N: int = 20
) -> Tuple[np.ndarray, np.ndarray]:
    """Explore-then-exploit: test each arm (round-robin) for N steps, then exploit best empirical arm.

    Returns:
        rewards: shape (T,) int array of 0/1 rewards
        choices: shape (T,) int array of chosen arm indices
    """
    K = len(mab)
    if N < K:
        pass

    counts = np.zeros(K, dtype=int)
    successes = np.zeros(K, dtype=int)
    rewards = []
    choices = []

    # exploration
    for t in range(N):
        a = t % K
        r = mab[a].sample()
        rewards.append(r)
        choices.append(a)
        counts[a] += 1
        successes[a] += r

    # exploitation
    empirical = successes / np.maximum(1, counts)
    best = int(np.argmax(empirical))
    for _ in range(N, T):
        r = mab[best].sample()
        rewards.append(r)
        choices.append(best)

    return np.array(rewards, dtype=int), np.array(choices, dtype=int)


def ucb(mab: List[ArmBernoulli], T: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    """UCB1 policy (Auer et al. 2002)."""
    K = len(mab)
    counts = np.zeros(K, dtype=int)
    successes = np.zeros(K, dtype=int)
    rewards: list[int] = []
    choices: list[int] = []

    for a in range(K):
        r = mab[a].sample()
        rewards.append(r)
        choices.append(a)
        counts[a] += 1
        successes[a] += r

    for t in range(K, T):
        t1 = t + 1
        empirical = successes / counts
        bonus = np.sqrt(2.0 * np.log(t1) / counts)
        ucb_vals = empirical + bonus
        a = int(np.argmax(ucb_vals))
        r = mab[a].sample()
        rewards.append(r)
        choices.append(a)
        counts[a] += 1
        successes[a] += r

    return np.array(rewards, dtype=int), np.array(choices, dtype=int)


def thompson(
    mab: List[ArmBernoulli], T: int = 100, alpha0: float = 1.0, beta0: float = 1.0
) -> Tuple[np.ndarray, np.ndarray]:
    """Thompson Sampling for Bernoulli arms with Beta prior."""
    K = len(mab)
    alpha = np.full(K, float(alpha0))
    beta = np.full(K, float(beta0))
    rewards: list[int] = []
    choices: list[int] = []

    rng = np.random.RandomState(0)

    for _ in range(T):
        samples = rng.beta(alpha, beta)
        a = int(np.argmax(samples))
        r = mab[a].sample()
        rewards.append(r)
        choices.append(a)
        alpha[a] += r
        beta[a] += 1 - r

    return np.array(rewards, dtype=int), np.array(choices, dtype=int)

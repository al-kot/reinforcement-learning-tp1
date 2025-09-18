import pytest
import numpy as np

from multiarmbandits.arms import ArmBernoulli
from multiarmbandits import algos

@pytest.fixture
def simple_means():
    return np.array([0.2, 0.8, 0.5])

@pytest.fixture
def mab(simple_means):
    rng = np.random.RandomState(0)
    return [ArmBernoulli(float(p), seed=int(rng.randint(0, 1_000_000))) for p in simple_means]

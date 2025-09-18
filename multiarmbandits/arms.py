from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from typing import Optional


@dataclass
class ArmBernoulli:
    """Bernoulli arm with internal RNG for reproducible sampling."""

    p: float
    seed: Optional[int] = None

    def __post_init__(self) -> None:
        self._rng = np.random.RandomState(self.seed)

    @property
    def mean(self) -> float:
        return float(self.p)

    def sample(self) -> int:
        """Return 1 for success, 0 for failure (Bernoulli trial)."""
        return int(self._rng.rand() < self.p)

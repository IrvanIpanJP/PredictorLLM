"""
Implements decay logic for importance and recency scores over time.
"""

import numpy as np
from typing import Tuple


class ExponentialDecay:
    """
    Applies exponential decay to importance and recency scores as time (delta) progresses.
    """

    def __init__(
        self,
        recency_factor: float = 10.0,
        importance_factor: float = 0.988
    ) -> None:
        """
        Initialize an ExponentialDecay object.

        Args:
            recency_factor (float, optional): Determines how quickly recency decays
                (larger = slower decay). Defaults to 10.0.
            importance_factor (float, optional): Multiplicative factor applied
                each step to importance (smaller < 1.0 => faster decay). Defaults to 0.988.
        """
        self.recency_factor = recency_factor
        self.importance_factor = importance_factor

    def __call__(
        self,
        importance_score: float,
        delta: float
    ) -> Tuple[float, float, float]:
        """
        Decay the importance_score and update the recency score as time passes.

        Args:
            importance_score (float): Current importance score.
            delta (float): The count of steps or 'days' since last significant use.

        Returns:
            Tuple[float, float, float]:
                (new_recency_score, new_importance_score, updated_delta)
        """
        updated_delta = delta + 1
        new_recency_score = np.exp(-(updated_delta / self.recency_factor))
        new_importance_score = importance_score * self.importance_factor

        return new_recency_score, new_importance_score, updated_delta

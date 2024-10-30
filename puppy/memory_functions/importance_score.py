"""
Defines initialization strategies for importance scores and a helper
to retrieve the appropriate strategy based on configuration.
"""

import numpy as np
from abc import ABC, abstractmethod


class ImportanceScoreInitialization(ABC):
    """
    Abstract base class for importance score initialization strategies.
    """

    @abstractmethod
    def __call__(self) -> float:
        """
        Produce an initial importance score (float).
        """
        pass


def get_importance_score_initialization_func(
    strategy_type: str,
    memory_layer: str
) -> ImportanceScoreInitialization:
    """
    Return an appropriate ImportanceScoreInitialization object
    based on the strategy type and memory layer.

    Args:
        strategy_type (str): e.g., "sample".
        memory_layer (str): e.g., "short", "mid", "long", "reflection".

    Returns:
        ImportanceScoreInitialization: A subclass instance for that layer.

    Raises:
        ValueError: If the strategy_type or memory_layer is invalid.
    """
    match strategy_type:
        case "sample":
            match memory_layer:
                case "short":
                    return ISampleInitializationShort()
                case "mid":
                    return ISampleInitializationMid()
                case "long":
                    return ISampleInitializationLong()
                case "reflection":
                    # reflection could reuse the "long" logic, for example
                    return ISampleInitializationLong()
                case _:
                    raise ValueError(f"Invalid memory layer type: {memory_layer}")
        case _:
            raise ValueError(f"Invalid importance score initialization type: {strategy_type}")


class ISampleInitializationShort(ImportanceScoreInitialization):
    """
    Probability-based importance score initialization for the short layer.
    """

    def __call__(self) -> float:
        """
        Samples an importance score from a discrete distribution with certain probabilities.
        """
        probabilities = [0.5, 0.45, 0.05]
        scores = [50.0, 70.0, 90.0]
        return float(np.random.choice(scores, p=probabilities))


class ISampleInitializationMid(ImportanceScoreInitialization):
    """
    Probability-based importance score initialization for the mid layer.
    """

    def __call__(self) -> float:
        """
        Samples an importance score from a discrete distribution with certain probabilities.
        """
        probabilities = [0.05, 0.8, 0.15]
        scores = [40.0, 60.0, 80.0]
        return float(np.random.choice(scores, p=probabilities))


class ISampleInitializationLong(ImportanceScoreInitialization):
    """
    Probability-based importance score initialization for the long (and reflection) layer.
    """

    def __call__(self) -> float:
        """
        Samples an importance score from a discrete distribution with certain probabilities.
        """
        probabilities = [0.05, 0.15, 0.8]
        scores = [40.0, 60.0, 80.0]
        return float(np.random.choice(scores, p=probabilities))

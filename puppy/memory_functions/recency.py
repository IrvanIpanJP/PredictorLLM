"""
Defines how recency scores may be initialized, plus any additional logic
that may be needed.
"""

class RConstantInitialization:
    """
    Initializes a recency score with a constant value (e.g., 1.0).
    """

    def __call__(self) -> float:
        """
        Return a fixed default recency score.
        """
        return 1.0

"""
Implements logic for how accessing (or referencing) a memory
record affects its importance score over time.
"""

class LinearImportanceScoreChange:
    """
    Increases importance score in a linear fashion based on access frequency.
    """

    def __call__(
        self,
        access_counter: int,
        importance_score: float
    ) -> float:
        """
        Update importance_score linearly with the number of accesses.

        Args:
            access_counter (int): Number of times the record has been accessed
                or received positive feedback.
            importance_score (float): Current importance score.

        Returns:
            float: The updated importance score.
        """
        return importance_score + (access_counter * 5)

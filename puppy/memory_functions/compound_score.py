"""
Defines classes and functions for combining recency and importance scores,
and merging them with similarity scores.
"""

class LinearCompoundScore:
    """
    Provides linear combinations of different scores (recency, importance, similarity).
    """

    def recency_and_importance_score(
        self,
        recency_score: float,
        importance_score: float
    ) -> float:
        """
        Combine recency_score and importance_score linearly.

        The importance_score is capped at 100 before combining, then scaled by 1/100.

        Args:
            recency_score (float): The recency component, typically between 0 and 1.
            importance_score (float): The importance score, typically up to 100.

        Returns:
            float: The combined (recency + scaled importance) score.
        """
        capped_importance = min(importance_score, 100)
        return recency_score + capped_importance / 100

    def merge_score(
        self,
        similarity_score: float,
        recency_and_importance: float
    ) -> float:
        """
        Merge embedding similarity with a combined recency-and-importance score
        via simple addition.

        Args:
            similarity_score (float): The cosine-similarity or dot-product measure.
            recency_and_importance (float): A previously calculated score that
                combines recency and importance.

        Returns:
            float: The merged score.
        """
        return similarity_score + recency_and_importance

from enum import Enum


class RunMode(Enum):
    """
    Indicates whether the agent is in training mode or testing mode.
    """
    Train = 0
    Test = 1

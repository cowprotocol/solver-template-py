"""Execution Plan Coordinates"""
from typing import Optional


class ExecPlanCoords:
    """The position coordinates of the uniswap in the execution plan.

    The position is defined by a pair of integers:
        * Id of the sequence.
        * Position within that sequence.
    """

    def __init__(self, sequence: Optional[int], position: Optional[int]):
        self.sequence = sequence
        self.position = position

    def as_tuple(self) -> tuple[Optional[int], Optional[int]]:
        """Returns tuple (sequence, position)"""
        return self.sequence, self.position

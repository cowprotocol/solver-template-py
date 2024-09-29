"""Execution Plan Coordinates"""


class ExecPlanCoords:
    """The position coordinates of the AMM in the execution plan.

    The position is defined by a pair of integers:
        * Id of the sequence.
        * Position within that sequence.
    """

    # pylint: disable=too-few-public-methods

    def __init__(self, sequence: int, position: int):
        self.sequence = sequence
        self.position = position

    def as_dict(self) -> dict[str, str]:
        """returns string dict of class"""
        return {
            "sequence": str(self.sequence),
            "position": str(self.position),
        }

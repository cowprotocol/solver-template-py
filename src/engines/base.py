"""
Base protocol for solver engines.
"""

from typing import Protocol
from src.domain.auction import Auction
from src.domain.solution import Solutions


class SolverEngine(Protocol):
    """Protocol for solver engines."""

    async def solve(self, auction: Auction) -> Solutions:
        """
        Solve an auction and return solutions.

        Args:
            auction: The auction to solve

        Returns:
            Solutions object containing the solver's solutions
        """
        ...

"""
Solution builder for creating valid settlement solutions.

Constructs the final solution with trades, interactions, and prices.
"""

from typing import List, Dict
from src.domain.solution import Solution, Trade, Interaction


class SolutionBuilder:
    """
    Builds valid solutions for the settlement contract.

    Formats trades, interactions, and prices according to the protocol.
    """

    def __init__(self, solution_gas_offset: int = 50000):
        self.solution_gas_offset = solution_gas_offset
        self.logger = __import__("logging").getLogger(__name__)

    def build_solution(
        self,
        auction_id: str,
        trades: List[Trade],
        interactions: List[Interaction],
        prices: Dict[str, int],
        order_index: int = 0,
    ) -> Solution:
        """
        Build a complete solution.

        Args:
            auction_id: Auction identifier
            trades: List of executed trades
            interactions: List of AMM interactions
            prices: Clearing prices for all tokens
            order_index: Index of the order being processed (for ID generation)

        Returns:
            Complete Solution object
        """
        # Convert prices from int to str as required by Solution model
        prices_str = {token: str(price) for token, price in prices.items()}

        # Calculate solution ID following Rust implementation pattern
        solution_id = self._generate_solution_id(auction_id, order_index)

        # Build the solution
        solution = Solution(
            id=solution_id, trades=trades, interactions=interactions, prices=prices_str
        )

        # Log solution details
        self.logger.info(
            f"Built solution {solution_id} with {len(trades)} trades, "
            f"{len(interactions)} interactions, {len(prices)} token prices"
        )

        return solution

    def _generate_solution_id(self, auction_id: str, order_index: int = 0) -> int:
        """
        Generate a unique solution ID following Rust implementation pattern.

        In the Rust implementation, the ID is generated as the order index (i as u64).
        This ensures consistent behavior with the reference implementation.

        Args:
            auction_id: Auction identifier (for logging)
            order_index: Index of the order being processed (0-based)

        Returns:
            Integer solution ID (u64 in Rust)
        """
        # Follow Rust implementation: use order index as ID
        # In Rust: .with_id(solution::Id(i as u64))
        return order_index

    def validate_solution(self, solution: Solution) -> bool:
        """
        Validate that a solution is well-formed.

        Args:
            solution: Solution to validate

        Returns:
            True if valid, False otherwise
        """
        # Check basic requirements
        if not solution.prices:
            self.logger.error("Solution has no prices")
            return False

        # Check all trades have valid order UIDs
        for trade in solution.trades:
            if not trade.order:
                self.logger.error("Trade missing order UID")
                return False
            if not trade.executed_amount:
                self.logger.error("Trade missing executed amount")
                return False

        # Validate interactions
        for interaction in solution.interactions:
            if not interaction.target:
                self.logger.error("Interaction missing target")
                return False
            if not interaction.call_data:
                self.logger.error("Interaction missing call data")
                return False

        return True

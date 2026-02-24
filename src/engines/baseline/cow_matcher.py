"""
Coincidence of Wants (CoW) matcher for direct order matching.

Finds and matches orders that can be settled directly against each other
without using AMM liquidity.
"""

from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict
from src.domain.order import Order
from src.domain.solution import Trade


@dataclass
class OrderMatch:
    """Represents a match between two orders."""

    buy_order: Order
    sell_order: Order
    execution_price: int  # Price in wei
    traded_amount: int  # Amount traded in sell token


class CowMatcher:
    """
    Matches orders directly without using AMM liquidity.

    Implements order matching logic similar to the Rust baseline solver.
    """

    def __init__(self):
        self.logger = __import__("logging").getLogger(__name__)

    def find_matches(self, orders: List[Order]) -> List[Trade]:
        """
        Find direct matches between buy and sell orders.

        Args:
            orders: List of orders to match

        Returns:
            List of Trade objects for matched orders
        """
        if not orders:
            return []

        # Group orders by token pairs
        order_book = self._build_order_book(orders)

        trades = []
        matched_orders = set()

        # Find matches for each token pair
        for (sell_token, buy_token), pair_orders in order_book.items():
            # Look for opposite direction orders
            opposite_key = (buy_token, sell_token)

            if opposite_key not in order_book:
                continue

            opposite_orders = order_book[opposite_key]

            # Match orders in this pair
            matches = self._match_order_pair(
                pair_orders, opposite_orders, matched_orders
            )

            # Convert matches to trades
            for match in matches:
                trades.extend(self._match_to_trades(match))

        self.logger.info(f"Found {len(trades)} CoW trades from {len(orders)} orders")
        return trades

    def _build_order_book(
        self, orders: List[Order]
    ) -> Dict[Tuple[str, str], List[Order]]:
        """
        Group orders by (sell_token, buy_token) pairs.

        Args:
            orders: List of orders to group

        Returns:
            Dictionary mapping token pairs to orders
        """
        order_book = defaultdict(list)

        for order in orders:
            # Normalize addresses to lowercase for comparison
            sell_token = order.sell_token.lower()
            buy_token = order.buy_token.lower()

            # Only include fillable orders
            if self._is_fillable(order):
                key = (sell_token, buy_token)
                order_book[key].append(order)

        # Sort orders by price for better matching
        for orders in order_book.values():
            orders.sort(key=lambda o: self._get_order_price(o), reverse=True)

        return dict(order_book)

    def _match_order_pair(
        self, orders_a: List[Order], orders_b: List[Order], matched_orders: set
    ) -> List[OrderMatch]:
        """
        Match orders between two opposite direction lists.

        Args:
            orders_a: Orders selling token A for token B
            orders_b: Orders selling token B for token A
            matched_orders: Set of already matched order UIDs

        Returns:
            List of order matches
        """
        matches = []

        for order_a in orders_a:
            if order_a.uid in matched_orders:
                continue

            for order_b in orders_b:
                if order_b.uid in matched_orders:
                    continue

                # Check if orders can match
                if self._can_match(order_a, order_b):
                    match = self._create_match(order_a, order_b)
                    if match:
                        matches.append(match)
                        matched_orders.add(order_a.uid)
                        matched_orders.add(order_b.uid)
                        break  # Move to next order_a

        return matches

    def _can_match(self, order_a: Order, order_b: Order) -> bool:
        """
        Check if two orders can be matched.

        Orders can match if:
        1. They trade opposite token pairs
        2. Their limit prices overlap
        3. Both are still fillable

        Args:
            order_a: First order
            order_b: Second order

        Returns:
            True if orders can be matched
        """
        # Check token pairs are opposite
        if not (
            order_a.sell_token.lower() == order_b.buy_token.lower()
            and order_a.buy_token.lower() == order_b.sell_token.lower()
        ):
            return False

        # Calculate limit prices
        price_a = self._get_order_price(order_a)
        price_b = self._get_order_price(order_b)

        # For opposite orders, prices should be reciprocal and overlap
        # order_a sells token X for Y at price P
        # order_b sells token Y for X at price Q
        # They match if P * Q >= 1 (accounting for decimals)

        # TODO: Simplified check - in production, consider decimals
        return price_a > 0 and price_b > 0

    def _create_match(self, order_a: Order, order_b: Order) -> Optional[OrderMatch]:
        """
        Create a match between two orders.

        Args:
            order_a: First order
            order_b: Second order

        Returns:
            OrderMatch if successful, None otherwise
        """
        # Determine which is buy and which is sell from perspective of order_a.sell_token
        if order_a.kind == "sell":
            sell_order = order_a
            buy_order = order_b
        else:
            sell_order = order_b
            buy_order = order_a

        # Calculate traded amounts
        # TODO: This is simplified - real implementation needs to handle partial fills
        traded_amount = min(int(sell_order.sell_amount), int(buy_order.buy_amount))

        if traded_amount <= 0:
            return None

        # Calculate execution price (in terms of sell token)
        execution_price = traded_amount

        return OrderMatch(
            buy_order=buy_order,
            sell_order=sell_order,
            execution_price=execution_price,
            traded_amount=traded_amount,
        )

    def _match_to_trades(self, match: OrderMatch) -> List[Trade]:
        """
        Convert an order match to Trade objects.

        Args:
            match: Order match to convert

        Returns:
            List of trades (one per order in the match)
        """
        trades = []

        # Trade for sell order (kind must be "fulfillment" for regular orders)
        trades.append(
            Trade(
                kind="fulfillment",
                order=match.sell_order.uid,
                executed_amount=str(match.traded_amount),
            )
        )

        # Trade for buy order (kind must be "fulfillment" for regular orders)
        trades.append(
            Trade(
                kind="fulfillment",
                order=match.buy_order.uid,
                executed_amount=str(match.traded_amount),
            )
        )

        return trades

    def _is_fillable(self, order: Order) -> bool:
        """Check if order can be filled."""
        return int(order.sell_amount) > 0 and int(order.buy_amount) > 0

    def _get_order_price(self, order: Order) -> float:
        """
        Get order limit price.

        Price = buy_amount / sell_amount
        """
        sell_amount = int(order.sell_amount)
        buy_amount = int(order.buy_amount)

        if sell_amount == 0:
            return 0

        return buy_amount / sell_amount

"""
Price finder for calculating clearing prices.

Determines uniform clearing prices for all tokens based on
executed trades and reference prices.

Based on the Rust baseline solver implementation.
"""

from typing import Dict, List, Optional, TYPE_CHECKING
from decimal import Decimal
from collections import defaultdict
import logging

import networkx as nx

from src.domain.solution import Trade

if TYPE_CHECKING:
    from src.domain.order import Order


class PriceFinder:
    """
    Calculates uniform clearing prices for settlements.

    Implements price finding logic similar to the Rust baseline solver,
    ensuring all trades clear at consistent prices.
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def find_clearing_prices(
        self,
        trades: List[Trade],
        reference_prices: Optional[Dict[str, int]] = None,
        base_token: Optional[str] = None,
        orders: Optional[Dict[str, "Order"]] = None,
    ) -> Dict[str, int]:
        """
        Calculate uniform clearing prices for all tokens.

        Args:
            trades: List of executed trades
            reference_prices: Optional reference prices from auction
            base_token: Optional base token for price normalization (e.g., WETH)
            orders: Optional dict mapping order UIDs to Order objects for token lookup

        Returns:
            Dictionary mapping token addresses to prices in wei
        """
        if not trades and not reference_prices:
            return {}

        # Start with reference prices if available
        prices = {}
        if reference_prices:
            prices = {k.lower(): int(v) for k, v in reference_prices.items()}

        # If no trades, return reference prices
        if not trades:
            return prices

        # Build price graph from trades
        if nx:
            price_graph = self._build_price_graph(trades, orders)

            # Find connected components
            components = list(nx.connected_components(price_graph.to_undirected()))

            # Process each component
            for component in components:
                # Find anchor token (prefer base token or highest liquidity)
                anchor = self._find_anchor_token(component, base_token, prices)

                if anchor and anchor in prices:
                    # Propagate prices from anchor
                    component_prices = self._propagate_prices(
                        price_graph, anchor, prices[anchor]
                    )
                    prices.update(component_prices)
                else:
                    # No anchor, use relative pricing
                    component_prices = self._calculate_relative_prices(
                        price_graph, component, trades
                    )
                    prices.update(component_prices)
        else:
            # No trades, use reference prices only
            pass

        # Normalize prices to ensure they're positive integers
        prices = self._normalize_prices(prices)

        self.logger.info(f"Calculated prices for {len(prices)} tokens")
        return prices

    def _build_price_graph(self, trades: List[Trade], orders: Optional[Dict[str, "Order"]] = None):
        """
        Build a directed graph representing price relationships.

        Each edge represents a trade and stores the exchange rate.

        Args:
            trades: List of Trade objects (fulfillment format with order UID)
            orders: Optional dict mapping order UIDs to Order objects
        """

        graph = nx.DiGraph()

        for trade in trades:
            # Look up order details from orders dict
            order_uid = trade.order
            if orders and order_uid in orders:
                order = orders[order_uid]
                sell_token = order.sell_token.lower()
                buy_token = order.buy_token.lower()
                sell_amount = int(order.sell_amount)
                buy_amount = int(order.buy_amount)
            else:
                # Skip if we can't find order details
                self.logger.warning(f"Order {order_uid[:20]}... not found in orders dict")
                continue

            if sell_amount > 0:
                # Add edge with exchange rate
                rate = Decimal(buy_amount) / Decimal(sell_amount)

                # Add bidirectional edges with rates
                graph.add_edge(
                    sell_token, buy_token, rate=rate, trade_id=order_uid
                )
                graph.add_edge(
                    buy_token,
                    sell_token,
                    rate=Decimal(1) / rate,
                    trade_id=order_uid,
                )

        return graph

    def _find_anchor_token(
        self, component: set, base_token: Optional[str], existing_prices: Dict[str, int]
    ) -> Optional[str]:
        """
        Find the best anchor token for price propagation.

        Priority:
        1. Base token if in component
        2. Token with existing price
        3. Token with most connections
        """
        # Check if base token is in component
        if base_token and base_token.lower() in component:
            return base_token.lower()

        # Check for tokens with existing prices
        for token in component:
            if token in existing_prices:
                return token

        # Return token with most connections (highest liquidity)
        # This would need the graph to determine connections
        return next(iter(component)) if component else None

    def _propagate_prices(
        self, graph, anchor: str, anchor_price: int
    ) -> Dict[str, int]:
        """
        Propagate prices from anchor token using exchange rates.

        Uses BFS to propagate prices through the graph.
        """
        prices = {anchor: anchor_price}
        visited = {anchor}
        queue = [anchor]

        while queue:
            current = queue.pop(0)
            current_price = prices[current]

            # Check all neighbors
            for neighbor in graph.neighbors(current):
                if neighbor not in visited:
                    # Calculate price based on exchange rate
                    edge_data = graph.get_edge_data(current, neighbor)
                    rate = edge_data["rate"]

                    neighbor_price = int(Decimal(current_price) * rate)
                    prices[neighbor] = neighbor_price

                    visited.add(neighbor)
                    queue.append(neighbor)

        return prices

    def _calculate_relative_prices(
        self, graph, component: set, trades: List[Trade]
    ) -> Dict[str, int]:
        """
        Calculate relative prices when no anchor is available.

        Sets one token to a base price and calculates others relative to it.
        """
        # Pick arbitrary token as base
        base_token = next(iter(component))
        base_price = 10**18  # 1 token in wei

        # Use propagation from this base
        return self._propagate_prices(graph, base_token, base_price)

    def _normalize_prices(self, prices: Dict[str, int]) -> Dict[str, int]:
        """
        Normalize prices to ensure they're positive integers.

        Also handles very small or very large prices.
        """
        if not prices:
            return {}

        normalized = {}

        for token, price in prices.items():
            # Ensure price is positive
            if price <= 0:
                price = 1

            # Cap extremely large prices
            max_price = 10**30  # Maximum reasonable price
            if price > max_price:
                price = max_price

            normalized[token] = int(price)

        return normalized

"""
Baseline solver engine implementation for CoW Protocol.

Based on the Rust implementation in cowprotocol-services/crates/solvers.
"""

import logging
from typing import List, Set, Dict, Optional, Tuple

from src.domain.auction import Auction
from src.domain.solution import Solutions, Trade, Interaction
from src.domain.order import Order

from src.engines.baseline.cow_matcher import CowMatcher
from src.engines.baseline.path_finder import PathFinder
from src.engines.baseline.solution_builder import SolutionBuilder

from src.engines.baseline.pools import PoolHandler
from src.engines.baseline.interactions import InteractionEncoder
from src.engines.baseline.price_finder import PriceFinder


class BaselineEngine:
    """
    Baseline solver engine implementation for CoW Protocol.

    Implements a multi-phase solving approach:
    1. Find CoW (Coincidence of Wants) matches between orders
    2. Route remaining orders through AMM pools
    3. Calculate clearing prices
    4. Build settlement solution
    """

    def __init__(
        self,
        weth_address: str,
        base_tokens: List[str],
        max_hops: int = 2,
        max_partial_attempts: int = 5,
        solution_gas_offset: int = 50000,
        native_token_price_estimation_amount: str = "1000000000000000000",
    ):
        """Initialize baseline solver with configuration."""
        self.logger = logging.getLogger(__name__)

        # Store configuration
        self.weth_address = weth_address
        self.base_tokens: Set[str] = set(base_tokens)
        self.max_hops = max_hops
        self.max_partial_attempts = max_partial_attempts
        self.solution_gas_offset = solution_gas_offset
        self.native_token_price_estimation_amount = int(
            native_token_price_estimation_amount
        )

        # Add WETH to base tokens if not present
        self.base_tokens.add(self.weth_address)

        # Initialize components
        self.cow_matcher = CowMatcher()
        self.path_finder = PathFinder(max_hops=max_hops)
        self.solution_builder = SolutionBuilder(solution_gas_offset=solution_gas_offset)
        self.pool_handler = PoolHandler()
        self.interaction_encoder = InteractionEncoder()
        self.price_finder = PriceFinder()

        self.logger.info(
            f"Initialized BaselineEngine: "
            f"chain={weth_address}, "
            f"base_tokens={len(self.base_tokens)}, "
            f"max_hops={max_hops}"
        )

    async def solve(self, auction: Auction) -> Solutions:
        """
        Solve auction using baseline algorithm.

        Steps:
        1. Find direct CoW matches
        2. Parse and prepare AMM liquidity
        3. Route remaining orders through AMMs
        4. Calculate clearing prices
        5. Build final solution with interactions
        """
        if not self._validate_auction(auction):
            return Solutions(solutions=[])

        try:
            # Phase 1: Find CoW matches
            cow_trades = self.cow_matcher.find_matches(auction.orders)
            self.logger.info(f"Found {len(cow_trades)} CoW trades")

            # Phase 2: Parse liquidity and build graph
            pools = {}
            graph = None

            if auction.liquidity:
                pools = self.pool_handler.parse_liquidity(auction.liquidity)
                graph = self.path_finder.build_graph(auction.liquidity)
                self.logger.info(f"Parsed {len(pools)} pools")

            # Phase 3: Route remaining orders through AMMs
            amm_trades = []
            interactions = []

            if graph and pools:
                # Get orders that weren't matched in CoW
                remaining_orders = self._get_unmatched_orders(
                    auction.orders, cow_trades
                )

                for order in remaining_orders:
                    # Try to route through AMMs
                    result = self._route_order_through_amms(order, graph, pools)

                    if result:
                        trades, order_interactions = result
                        amm_trades.extend(trades)
                        interactions.extend(order_interactions)

            # Phase 4: Calculate clearing prices
            all_trades = cow_trades + amm_trades

            # Get reference prices from auction
            reference_prices = self._extract_reference_prices(auction.tokens)

            # Create orders dict for price finder to look up token info
            orders_dict = {order.uid: order for order in auction.orders}

            # Calculate final clearing prices
            prices = self.price_finder.find_clearing_prices(
                trades=all_trades,
                reference_prices=reference_prices,
                base_token=self.weth_address,
                orders=orders_dict,
            )

            # Ensure all tokens have prices
            for token_addr in auction.tokens.keys():
                if token_addr.lower() not in prices:
                    prices[token_addr.lower()] = 10**18  # Default price

            # Phase 5: Build solution
            solution = self.solution_builder.build_solution(
                auction_id=auction.id,
                trades=all_trades,
                interactions=interactions,
                prices=prices,
            )

            # Validate solution
            if self.solution_builder.validate_solution(solution):
                self.logger.info(
                    f"Created solution with {len(solution.trades)} trades, "
                    f"{len(solution.interactions)} interactions"
                )
                return Solutions(solutions=[solution])
            else:
                self.logger.error("❌ Solution validation failed")
                return Solutions(solutions=[])

        except Exception as e:
            self.logger.error(f"Error solving auction: {e}", exc_info=True)
            return Solutions(solutions=[])

    def _route_order_through_amms(
        self, order: Order, graph, pools: Dict
    ) -> Optional[Tuple[List[Trade], List[Interaction]]]:
        """
        Route an order through AMM pools.

        Args:
            order: Order to route
            graph: Liquidity graph
            pools: Pool handler with parsed pools

        Returns:
            Tuple of (trades, interactions) or None if routing fails
        """
        sell_token = order.sell_token.lower()
        buy_token = order.buy_token.lower()

        # Find path
        path = self.path_finder.find_path(
            sell_token=sell_token,
            buy_token=buy_token,
            graph=graph,
            base_tokens=self.base_tokens,
        )

        if not path:
            return None

        # Execute swaps along path
        trades = []
        interactions = []
        current_amount = int(order.sell_amount)

        for i in range(len(path.tokens) - 1):
            token_in = path.tokens[i]
            token_out = path.tokens[i + 1]
            pool_id = path.pools[i]

            # Execute swap in pool
            result = self.pool_handler.execute_swap(
                pool_id=pool_id,
                token_in=token_in,
                token_out=token_out,
                amount_in=current_amount,
            )

            if not result:
                return None

            amount_out, gas_used = result

            # Create interaction for this swap
            pool = self.pool_handler.get_pool_by_id(pool_id)
            if pool:
                interaction = self.interaction_encoder.encode_uniswap_v2_swap(
                    pool_address=pool.address,
                    token_in=token_in,
                    token_out=token_out,
                    amount_in=current_amount,
                    amount_out_min=int(amount_out * 0.97),
                )
                interactions.append(interaction)

            current_amount = amount_out

        executed_amount = (
            str(order.sell_amount) if order.kind == "sell" else str(current_amount)
        )

        trade = Trade(
            kind="fulfillment",
            order=order.uid,
            executed_amount=executed_amount,
        )
        trades.append(trade)

        return (trades, interactions)

    def _get_unmatched_orders(
        self, all_orders: List[Order], cow_trades: List[Trade]
    ) -> List[Order]:
        """Get orders that weren't matched in CoW."""
        matched_uids = {trade.order for trade in cow_trades}
        return [order for order in all_orders if order.uid not in matched_uids]

    def _extract_reference_prices(self, tokens: Dict) -> Dict[str, int]:
        """Extract reference prices from auction tokens."""
        prices = {}

        for token_addr, token_info in tokens.items():
            token_addr = token_addr.lower()

            # Try different price field names
            if (
                hasattr(token_info, "reference_price")
                and token_info.reference_price is not None
            ):
                prices[token_addr] = int(token_info.reference_price)
            elif (
                hasattr(token_info, "external_price")
                and token_info.external_price is not None
            ):
                # Convert decimal price to wei
                price_decimal = float(token_info.external_price)
                prices[token_addr] = int(price_decimal * 10**18)
            elif isinstance(token_info, dict):
                if (
                    "reference_price" in token_info
                    and token_info["reference_price"] is not None
                ):
                    prices[token_addr] = int(token_info["reference_price"])
                elif (
                    "external_price" in token_info
                    and token_info["external_price"] is not None
                ):
                    price_decimal = float(token_info["external_price"])
                    prices[token_addr] = int(price_decimal * 10**18)

            # Default price if not found
            if token_addr not in prices:
                prices[token_addr] = 10**18

        return prices

    def _validate_auction(self, auction: Auction) -> bool:
        """
        Validate the auction data.

        Args:
            auction: The auction to validate

        Returns:
            True if valid, False otherwise
        """
        if not auction.id:
            self.logger.error("Auction has no ID")
            return False

        if not auction.orders:
            self.logger.warning("Auction has no orders")
            # Still valid, just no orders to solve

        if not auction.tokens:
            self.logger.error("Auction has no tokens")
            return False

        return True

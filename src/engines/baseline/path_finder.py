"""
Path finding for order routing through AMM liquidity.

Builds a graph of available liquidity and finds optimal paths
for routing orders.
"""

from typing import List, Set, Optional, Dict, Tuple
import networkx as nx
from dataclasses import dataclass
from src.domain.liquidity import Liquidity
from src.utils.fee_conversion import fee_to_basis_points


@dataclass
class TradePath:
    """Represents a trading path through AMM pools."""

    tokens: List[str]  # Ordered list of tokens in path
    pools: List[str]  # Pool addresses for each hop
    estimated_gas: int  # Total gas estimate


class PathFinder:
    """
    Finds optimal trading paths through AMM liquidity.

    Implements BFS/Dijkstra with max_hops constraint.
    """

    def __init__(self, max_hops: int = 2):
        self.max_hops = max_hops
        self.logger = __import__("logging").getLogger(__name__)

    def build_graph(self, liquidity_sources: List[Liquidity]) -> nx.DiGraph:
        """
        Build a directed graph of available liquidity.

        Args:
            liquidity_sources: List of liquidity sources from auction

        Returns:
            Directed graph with tokens as nodes and pools as edges
        """
        graph = nx.DiGraph()

        for liquidity in liquidity_sources:
            # Parse liquidity based on kind
            if liquidity.kind == "constantProduct":
                self._add_constant_product_pool(graph, liquidity)
            elif liquidity.kind == "weightedProduct":
                self._add_weighted_product_pool(graph, liquidity)
            # Add other pool types as needed

        self.logger.info(
            f"Built liquidity graph with {graph.number_of_nodes()} tokens "
            f"and {graph.number_of_edges()} edges"
        )

        return graph

    def find_path(
        self, sell_token: str, buy_token: str, graph: nx.DiGraph, base_tokens: Set[str]
    ) -> Optional[TradePath]:
        """
        Find optimal trading path between two tokens.

        Args:
            sell_token: Source token address
            buy_token: Target token address
            graph: Liquidity graph
            base_tokens: Set of preferred intermediate tokens

        Returns:
            TradePath if found, None otherwise
        """
        sell_token = sell_token.lower()
        buy_token = buy_token.lower()

        # Try direct path first
        if graph.has_edge(sell_token, buy_token):
            edge_data = graph.get_edge_data(sell_token, buy_token)
            return TradePath(
                tokens=[sell_token, buy_token],
                pools=[edge_data["pool"]],
                estimated_gas=edge_data["gas"],
            )

        # Try paths through base tokens
        best_path = None
        best_gas = float("inf")

        for base_token in base_tokens:
            base_token = base_token.lower()

            if base_token in [sell_token, buy_token]:
                continue

            path = self._find_path_through_base(
                graph, sell_token, buy_token, base_token
            )

            if path and path.estimated_gas < best_gas:
                best_path = path
                best_gas = path.estimated_gas

        # If no path through base tokens, try any path within hop limit
        if not best_path:
            best_path = self._find_any_path(graph, sell_token, buy_token, self.max_hops)

        return best_path

    def _add_constant_product_pool(self, graph: nx.DiGraph, liquidity: Liquidity):
        """Add a Uniswap V2 style pool to the graph."""
        # Extract tokens from the liquidity data
        tokens = list(liquidity.tokens.keys())

        if len(tokens) != 2:
            return

        token0 = tokens[0].lower()
        token1 = tokens[1].lower()

        # Add bidirectional edges
        graph.add_edge(
            token0,
            token1,
            pool=liquidity.id,
            kind="constant_product",
            gas=int(liquidity.gas_estimate),
            fee=fee_to_basis_points(liquidity.fee),
        )
        graph.add_edge(
            token1,
            token0,
            pool=liquidity.id,
            kind="constant_product",
            gas=int(liquidity.gas_estimate),
            fee=fee_to_basis_points(liquidity.fee),
        )

    def _add_weighted_product_pool(self, graph: nx.DiGraph, liquidity: Liquidity):
        """Add a Balancer weighted pool to the graph."""
        # Weighted pools can have multiple tokens
        tokens = [t.lower() for t in liquidity.tokens.keys()]

        # Add edges between all token pairs
        for i, token_a in enumerate(tokens):
            for token_b in tokens[i + 1 :]:
                graph.add_edge(
                    token_a,
                    token_b,
                    pool=liquidity.id,
                    kind="weighted_product",
                    gas=int(liquidity.gas_estimate),
                    fee=fee_to_basis_points(liquidity.fee),
                )
                graph.add_edge(
                    token_b,
                    token_a,
                    pool=liquidity.id,
                    kind="weighted_product",
                    gas=int(liquidity.gas_estimate),
                    fee=fee_to_basis_points(liquidity.fee),
                )

    def _find_path_through_base(
        self, graph: nx.DiGraph, sell_token: str, buy_token: str, base_token: str
    ) -> Optional[TradePath]:
        """Find path through a specific base token."""
        # Check if path exists: sell -> base -> buy
        if graph.has_edge(sell_token, base_token) and graph.has_edge(
            base_token, buy_token
        ):

            edge1 = graph.get_edge_data(sell_token, base_token)
            edge2 = graph.get_edge_data(base_token, buy_token)

            return TradePath(
                tokens=[sell_token, base_token, buy_token],
                pools=[edge1["pool"], edge2["pool"]],
                estimated_gas=edge1["gas"] + edge2["gas"],
            )

        return None

    def _find_any_path(
        self, graph: nx.DiGraph, sell_token: str, buy_token: str, max_hops: int
    ) -> Optional[TradePath]:
        """Find any valid path within hop limit."""
        if not graph.has_node(sell_token) or not graph.has_node(buy_token):
            return None

        try:
            # Find all simple paths within hop limit
            paths = list(
                nx.all_simple_paths(
                    graph,
                    sell_token,
                    buy_token,
                    cutoff=max_hops + 1,  # cutoff is in nodes, not edges
                )
            )

            if not paths:
                return None

            # Find path with minimum gas cost
            best_path = None
            best_gas = float("inf")

            for path in paths:
                pools = []
                total_gas = 0

                for i in range(len(path) - 1):
                    edge_data = graph.get_edge_data(path[i], path[i + 1])
                    pools.append(edge_data["pool"])
                    total_gas += edge_data["gas"]

                if total_gas < best_gas:
                    best_path = TradePath(
                        tokens=path, pools=pools, estimated_gas=total_gas
                    )
                    best_gas = total_gas

            return best_path

        except nx.NetworkXNoPath:
            return None

"""
Pool handler for managing different AMM types.

Converts auction liquidity data into internal pool representations
and handles swap execution calculations.

Based on the Rust baseline solver implementation.
"""

from typing import Union, Optional, Dict, Tuple, List
from dataclasses import dataclass
from decimal import Decimal
import logging

from src.domain.liquidity import Liquidity
from src.utils.amm_math import UniswapV2, BalancerWeighted, CurveStable
from src.utils.fee_conversion import fee_to_basis_points


@dataclass
class ConstantProductPool:
    """Internal representation of a Uniswap V2 style pool."""

    address: str
    token0: str
    token1: str
    reserve0: int
    reserve1: int
    fee_bps: int  # Fee in basis points

    def get_amount_out(
        self, token_in: str, amount_in: int
    ) -> Optional[Tuple[str, int]]:
        """Calculate output amount for a swap."""
        if token_in == self.token0:
            amount_out = UniswapV2.get_amount_out(
                amount_in, self.reserve0, self.reserve1, self.fee_bps
            )
            return (self.token1, amount_out) if amount_out and amount_out > 0 else None
        elif token_in == self.token1:
            amount_out = UniswapV2.get_amount_out(
                amount_in, self.reserve1, self.reserve0, self.fee_bps
            )
            return (self.token0, amount_out) if amount_out and amount_out > 0 else None
        return None

    def get_amount_in(
        self, token_out: str, amount_out: int
    ) -> Optional[Tuple[str, int]]:
        """Calculate required input amount for desired output."""
        if token_out == self.token0:
            amount_in = UniswapV2.get_amount_in(
                amount_out, self.reserve1, self.reserve0, self.fee_bps
            )
            return (self.token1, amount_in) if amount_in and amount_in > 0 else None
        elif token_out == self.token1:
            amount_in = UniswapV2.get_amount_in(
                amount_out, self.reserve0, self.reserve1, self.fee_bps
            )
            return (self.token0, amount_in) if amount_in and amount_in > 0 else None
        return None


@dataclass
class WeightedProductPool:
    """Internal representation of a Balancer weighted pool."""

    address: str
    tokens: List[str]
    balances: List[int]
    weights: List[int]  # As percentages (e.g., 50 for 50%)
    fee_bps: int

    def get_amount_out(
        self, token_in: str, token_out: str, amount_in: int
    ) -> Optional[int]:
        """Calculate output amount for a weighted pool swap."""
        try:
            in_idx = self.tokens.index(token_in)
            out_idx = self.tokens.index(token_out)
        except ValueError:
            return None

        amount_out = BalancerWeighted.calc_out_given_in(
            balance_in=self.balances[in_idx],
            weight_in=self.weights[in_idx],
            balance_out=self.balances[out_idx],
            weight_out=self.weights[out_idx],
            amount_in=amount_in,
            fee_bps=self.fee_bps,
        )

        return amount_out if amount_out and amount_out > 0 else None

    def get_amount_in(
        self, token_out: str, token_in: str, amount_out: int
    ) -> Optional[int]:
        """Calculate required input amount for weighted pool."""
        try:
            in_idx = self.tokens.index(token_in)
            out_idx = self.tokens.index(token_out)
        except ValueError:
            return None

        amount_in = BalancerWeighted.calc_in_given_out(
            balance_in=self.balances[in_idx],
            weight_in=self.weights[in_idx],
            balance_out=self.balances[out_idx],
            weight_out=self.weights[out_idx],
            amount_out=amount_out,
            fee_bps=self.fee_bps,
        )

        return amount_in if amount_in and amount_in > 0 else None


@dataclass
class StablePool:
    """Internal representation of a Curve stable pool."""

    address: str
    tokens: List[str]
    balances: List[int]
    amplification: int
    fee_bps: int

    def get_amount_out(
        self, token_in: str, token_out: str, amount_in: int
    ) -> Optional[int]:
        """Calculate output for stable pool swap."""
        try:
            in_idx = self.tokens.index(token_in)
            out_idx = self.tokens.index(token_out)
        except ValueError:
            return None

        amount_out = CurveStable.get_amount_out(
            amount_in=amount_in,
            reserve_in=self.balances[in_idx],
            reserve_out=self.balances[out_idx],
            amplification=self.amplification,
            fee_bps=self.fee_bps,
        )

        return amount_out if amount_out and amount_out > 0 else None


class PoolHandler:
    """
    Manages pool conversions and swap calculations.

    Converts liquidity sources from auction data into internal pool
    representations and handles swap execution logic.
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.pools: Dict[
            str, Union[ConstantProductPool, WeightedProductPool, StablePool]
        ] = {}

    def parse_liquidity(self, liquidity_sources: List[Liquidity]) -> Dict[str, any]:
        """
        Parse liquidity sources into internal pool representations.

        Args:
            liquidity_sources: List of liquidity from auction

        Returns:
            Dictionary mapping pool IDs to pool objects
        """
        pools = {}

        for liquidity in liquidity_sources:
            pool = self._liquidity_to_pool(liquidity)
            if pool:
                pools[liquidity.id] = pool
                self.pools[liquidity.id] = pool

        self.logger.info(f"Parsed {len(pools)} pools from liquidity sources")
        return pools

    def _liquidity_to_pool(
        self, liquidity: Liquidity
    ) -> Optional[Union[ConstantProductPool, WeightedProductPool, StablePool]]:
        """
        Convert a liquidity source to internal pool representation.

        Args:
            liquidity: Liquidity source from auction

        Returns:
            Pool object or None if conversion fails
        """
        try:
            # Handle both camelCase and PascalCase
            kind = liquidity.kind.lower()
            if kind == "constantproduct":
                return self._parse_constant_product(liquidity)
            elif kind == "weightedproduct":
                return self._parse_weighted_product(liquidity)
            elif kind == "stable":
                return self._parse_stable_pool(liquidity)
            else:
                self.logger.warning(f"Unsupported liquidity kind: {liquidity.kind}")
                return None
        except Exception as e:
            self.logger.error(f"Error parsing liquidity {liquidity.id}: {e}")
            return None

    def _parse_constant_product(
        self, liquidity: Liquidity
    ) -> Optional[ConstantProductPool]:
        """Parse Uniswap V2 style pool."""
        tokens = list(liquidity.tokens.keys())
        if len(tokens) != 2:
            return None

        # Extract reserves from tokens field
        reserves = []
        for token in tokens:
            balance = liquidity.tokens[token]
            # Handle both string and dict formats
            if isinstance(balance, dict):
                reserves.append(int(balance.get("balance", 0)))
            else:
                reserves.append(int(balance))

        return ConstantProductPool(
            address=liquidity.address.lower(),
            token0=tokens[0].lower(),
            token1=tokens[1].lower(),
            reserve0=reserves[0],
            reserve1=reserves[1],
            fee_bps=fee_to_basis_points(liquidity.fee),  # Convert to basis points
        )

    def _parse_weighted_product(
        self, liquidity: Liquidity
    ) -> Optional[WeightedProductPool]:
        """Parse Balancer weighted pool."""
        tokens = []
        balances = []
        weights = []

        for token_addr, token_data in liquidity.tokens.items():
            tokens.append(token_addr.lower())

            # Handle different data formats
            if isinstance(token_data, dict):
                balances.append(int(token_data.get("balance", 0)))
                # Parse weight (can be decimal like "0.5" or percentage like "50")
                weight_str = token_data.get("weight", "0")
                weight_decimal = Decimal(weight_str)
                if weight_decimal <= 1:
                    weights.append(int(weight_decimal * 100))  # Convert to percentage
                else:
                    weights.append(int(weight_decimal))
            else:
                balances.append(int(token_data))
                weights.append(50)  # Default to equal weight

        return WeightedProductPool(
            address=liquidity.address.lower(),
            tokens=tokens,
            balances=balances,
            weights=weights,
            fee_bps=fee_to_basis_points(liquidity.fee),
        )

    def _parse_stable_pool(self, liquidity: Liquidity) -> Optional[StablePool]:
        """Parse Curve stable pool."""
        tokens = []
        balances = []

        for token_addr, balance in liquidity.tokens.items():
            tokens.append(token_addr.lower())
            if isinstance(balance, dict):
                balances.append(int(balance.get("balance", 0)))
            else:
                balances.append(int(balance))

        # Extract amplification parameter
        amplification = 100  # Default
        if hasattr(liquidity, "amplification"):
            amplification = int(liquidity.amplification)

        return StablePool(
            address=liquidity.address.lower(),
            tokens=tokens,
            balances=balances,
            amplification=amplification,
            fee_bps=fee_to_basis_points(liquidity.fee),
        )

    def execute_swap(
        self, pool_id: str, token_in: str, token_out: str, amount_in: int
    ) -> Optional[Tuple[int, int]]:  # Returns (amount_out, gas_used)
        """
        Execute a swap in a pool.

        Args:
            pool_id: Pool identifier
            token_in: Input token address
            token_out: Output token address
            amount_in: Input amount in wei

        Returns:
            Tuple of (output amount, gas used) or None if swap fails
        """
        pool = self.pools.get(pool_id)
        if not pool:
            return None

        token_in = token_in.lower()
        token_out = token_out.lower()

        # Calculate swap based on pool type
        if isinstance(pool, ConstantProductPool):
            result = pool.get_amount_out(token_in, amount_in)
            if result and result[0] == token_out:
                return (result[1], 110_000)  # Uniswap V2 gas

        elif isinstance(pool, WeightedProductPool):
            amount_out = pool.get_amount_out(token_in, token_out, amount_in)
            if amount_out:
                return (amount_out, 150_000)  # Balancer gas

        elif isinstance(pool, StablePool):
            amount_out = pool.get_amount_out(token_in, token_out, amount_in)
            if amount_out:
                return (amount_out, 175_000)  # Curve gas

        return None

    def get_pool_by_id(self, pool_id: str) -> Optional[any]:
        """Get pool by ID."""
        return self.pools.get(pool_id)

    def get_pools_for_pair(self, token_a: str, token_b: str) -> List[str]:
        """Get all pools that can trade between two tokens."""
        token_a = token_a.lower()
        token_b = token_b.lower()
        matching_pools = []

        for pool_id, pool in self.pools.items():
            if isinstance(pool, ConstantProductPool):
                if {pool.token0, pool.token1} == {token_a, token_b}:
                    matching_pools.append(pool_id)
            elif isinstance(pool, (WeightedProductPool, StablePool)):
                if token_a in pool.tokens and token_b in pool.tokens:
                    matching_pools.append(pool_id)

        return matching_pools

"""
Model containing BatchAuction which is what solvers operate on.
"""

from __future__ import annotations
import decimal
import logging
from decimal import Decimal
from typing import Any, Optional

from src.models.order import Order, OrdersSerializedType
from src.models.token import (
    Token,
    TokenInfo,
    select_token_with_highest_normalize_priority,
    TokenDict,
    TokenSerializedType,
)
from src.models.types import NumericType
from src.models.uniswap import Uniswap, UniswapsSerializedType
from src.util.enums import Chain


class BatchAuction:
    """Class to represent a batch auction."""

    def __init__(
        self,
        tokens: dict[Token, TokenInfo],
        orders: dict[str, Order],
        uniswaps: dict[str, Uniswap],
        ref_token: Token,
        prices: Optional[dict] = None,
        name: str = "batch_auction",
        metadata: Optional[dict] = None,
    ):
        """Initialize.
        Args:
            tokens: dict of tokens participating.
            orders: dict of Order objects.
            uniswaps: dict of Uniswap objects.
            ref_token: Reference Token object.
            prices: A dict of {token -> price}.
            name: Name of the batch auction instance.
            metadata: Some instance metadata.
        """
        self.name = name
        self.metadata = metadata if metadata else {}

        self._tokens = tokens
        self._orders = orders
        self._uniswaps = uniswaps

        # Store reference token and (previous) prices.
        self.ref_token = ref_token
        self.prices = (
            prices if prices else {ref_token: self._tokens[ref_token].external_price}
        )

    @classmethod
    def from_dict(cls, data: dict, name: str) -> BatchAuction:
        """Read a batch auction instance from a python dictionary.

        Args:
            data: Python dict to be read.
            name: Instance name.

        Returns:
            The instance.

        """
        for key in ["tokens", "orders"]:
            if key not in data:
                raise ValueError(f"Mandatory field '{key}' missing in instance data!")

        tokens = load_tokens(data["tokens"])
        orders = load_orders(data["orders"])

        uniswaps = load_amms(data.get("amms", {}))
        metadata = load_metadata(data.get("metadata", {}))
        prices = load_prices(data.get("prices", {}))

        ref_token = select_token_with_highest_normalize_priority(tokens)

        return cls(
            tokens,
            orders,
            uniswaps,
            ref_token,
            prices=prices,
            metadata=metadata,
            name=name,
        )

    ####################
    #  ACCESS METHODS  #
    ####################

    @property
    def tokens(self) -> list[Token]:
        """Access to (sorted) token list."""
        return sorted(self._tokens.keys())

    @property
    def orders(self) -> list[Order]:
        """Access to order list."""
        return list(self._orders.values())

    @property
    def uniswaps(self) -> list[Uniswap]:
        """Access to uniswap list."""
        return list(self._uniswaps.values())

    def token_info(self, token: Token) -> TokenInfo:
        """Get the token info for a specific token."""
        assert isinstance(token, Token)

        if token not in self.tokens:
            raise ValueError(f"Token <{token}> not in batch auction!")

        return self._tokens[token]

    @property
    def chain(self) -> Chain:
        """Return the blockchain on which the BatchAuction happens."""
        if self.ref_token is None:
            return Chain.UNKNOWN
        ref = self.ref_token.value.lower()
        if ref == "0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2":
            return Chain.MAINNET
        if ref == "0xe91d153e0b41518a2ce8dd3d7944fa863463a97d":
            return Chain.XDAI
        return Chain.UNKNOWN

    @property
    def default_ref_token_price(self) -> Decimal:
        """Price of reference token if not given explicitely.

        This price is chosen so that the price of one unit of
        a token with d_t=18 decimals that costs one unit of the
        reference token is p_t=10^18:

        a_t * p_t * 10^d_t = a_r * p_r * 10^d_r

        with:
            a_t/a_r = 1
            d_t = 18
            p_t = 10^18

        p_r = a_t/a_r * 10^18 * 10^18 / 10^d_r
        <-> p_r = 10^(2 * 18 - d_r)
        """
        return Decimal(10) ** (2 * 18 - self.token_info(self.ref_token).decimals)

    def solve(self) -> None:
        """Solve Batch"""

    #################################
    #  SOLUTION PROCESSING METHODS  #
    #################################

    def __str__(self) -> str:
        """Print batch auction data.

        Returns:
            The string representation.

        """
        output_str = "BATCH AUCTION:"

        output_str += f"\n=== TOKENS ({len(self.tokens)}) ==="
        for token in self.tokens:
            output_str += f"\n-- {token}"

        output_str += f"\n=== ORDERS ({len(self.orders)}) ==="
        for order in self.orders:
            output_str += f"\n{order}"

        output_str += f"\n=== UNISWAPS ({len(self.uniswaps)}) ==="
        for uni in self.uniswaps:
            output_str += f"\n{uni}"

        return output_str

    def __repr__(self) -> str:
        """Print batch auction data."""
        return self.name


def load_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """Store some basic metadata information."""
    metadata["scaling_factors"] = {
        Token(t): Decimal(f) for t, f in metadata.get("scaling_factors", {}).items()
    }

    return metadata


def load_prices(
    prices_serialized: dict[TokenSerializedType, NumericType]
) -> dict[Token, Decimal]:
    """Load token price information as dict of Token -> Decimal."""
    if not isinstance(prices_serialized, dict):
        raise ValueError(
            f"Prices must be given as dict, not {type(prices_serialized)}!"
        )
    return {Token(t): Decimal(p) for t, p in prices_serialized.items()}


def load_orders(orders_serialized: OrdersSerializedType) -> dict[str, Order]:
    """Load dict of orders as order pool_id -> order data.

    Args:
        orders_serialized: dict of order_id -> order data dict.

    Returns:
        A list of Order objects.
    """
    order_list = [
        Order.from_dict(order_id, data) for order_id, data in orders_serialized.items()
    ]
    result: dict[str, Order] = {}
    for order in order_list:
        if order.order_id in result:
            raise ValueError(f"Order pool_id <{order.order_id}> already exists!")
        result[order.order_id] = order
    return result


def load_amms(amms_serialized: UniswapsSerializedType) -> dict[str, Uniswap]:
    """Load list of AMMs.

    NOTE: Currently, the code only supports Uniswap-style AMMs, i.e.,
    constant-product pools with two tokens and equal weights.

    Args:
        amms_serialized: dict of pool_id -> AMM.

    Returns:
        A list of Uniswap objects.

    """
    amm_list = []
    for amm_id, amm_data in amms_serialized.items():
        amm = Uniswap.from_dict(amm_id, amm_data)
        if amm is not None:
            amm_list.append(amm)

    results: dict[str, Uniswap] = {}
    for uni in amm_list:
        if uni.pool_id in results:
            raise ValueError(f"Uniswap pool_id <{uni.pool_id}> already exists!")
        results[uni.pool_id] = uni

    return results


def load_tokens(tokens_serialized: dict) -> TokenDict:
    """Store tokens as sorted dictionary from Token -> token info.

    Args:
        tokens_serialized: list or dict of tokens.

    Returns:
        A dict of Token -> token info.

    """
    tokens_dict = {}
    for token_str, token_info in sorted(tokens_serialized.items()):
        token = Token(token_str)
        if token_info is None:
            token_info = {}
        else:
            for k, val in token_info.items():
                if val is None:
                    continue
                try:
                    # Convert to str first to avoid float-precision artifacts:
                    # Decimal(0.1)   -> Decimal('0.10000000000000000555...')
                    # Decimal('0.1') -> Decimal(0.1)
                    val = Decimal(str(val))
                except decimal.InvalidOperation:
                    pass
                token_info[k] = val
        if token in tokens_dict:
            logging.warning(f"Token <{token}> already exists!")
        tokens_dict[token] = TokenInfo(token, **token_info)

    return tokens_dict

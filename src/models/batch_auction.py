"""
Model containing BatchAuction which is what solvers operate on.
"""

from __future__ import annotations
import decimal
import itertools
import logging
from decimal import Decimal
from typing import Any, Optional

import inflection

from src.models.order import Order, OrdersSerializedType
from src.models.token import (
    Token,
    TokenBalance,
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
        orders: list[Order],
        uniswaps: list[Uniswap],
        ref_token: Token,
        prices: Optional[dict] = None,
        name: str = "batch_auction",
        metadata: Optional[dict] = None,
    ):
        """Initialize.

        Args:
            tokens: list of tokens participating.
            orders: list of Order objects.
            uniswaps: list of Uniswap objects.
            ref_token: Reference Token object.
            prices: A dict of {token -> price}.
            name: Name of the batch auction instance.
            metadata: Some instance metadata.

        """
        # Create attributes so that calling __repr__ on this instance
        # before the ctor exits does not raise.
        self._tokens: dict[Token, TokenInfo] = {}
        self._orders: dict[str, Order] = {}
        self._uniswaps: dict[str, Uniswap] = {}

        # Store list of participating tokens, orders, and uniswaps.
        self._add_tokens(tokens)
        self._add_orders(orders)
        self._add_uniswaps(uniswaps)

        # Store reference token and (previous) prices.
        self._ref_token = ref_token
        self.prices = prices if prices else {}

        self.validate()

        self.name = name
        self.metadata = metadata if metadata else {}

        # Init objective values.
        self.trading_volume = float("nan")
        self.trading_surplus = float("nan")

    @classmethod
    def from_dict(cls, inst_dict: dict, inst_name: str) -> BatchAuction:
        """Read a batch auction instance from a python dictionary.

        Args:
            inst_dict: Python dict to be read.
            inst_name: Instance name.

        Returns:
            The instance.

        """
        if not isinstance(inst_dict, dict):
            raise ValueError(f"Instance data must be a dict, not {type(inst_dict)}!")

        for key in ["tokens", "orders"]:
            if key not in inst_dict:
                raise ValueError(f"Mandatory field '{key}' missing in instance data!")

        tokens = load_tokens(inst_dict["tokens"])
        metadata = load_metadata(inst_dict.get("metadata", {}))
        prices = load_prices(inst_dict.get("prices", {}))
        orders = load_orders(inst_dict["orders"])
        uniswaps = load_amms(inst_dict.get("amms", {}))
        ref_token = select_ref_token(tokens)

        log_str = "Loaded tokens:"
        for token_info in tokens.values():
            log_str += f"\n{token_info}"
        logging.debug(log_str)

        log_str = "Loaded orders:"
        for order in orders:
            log_str += f"\n{order}"
        logging.debug(log_str)

        log_str = "Loaded uniswaps:"
        for uniswap in uniswaps:
            log_str += f"\n{uniswap}"
        logging.debug(log_str)

        return cls(
            tokens,
            orders,
            uniswaps,
            ref_token,
            prices=prices,
            metadata=metadata,
            name=inst_name,
        )

    #######################################
    #  HELPER METHODS FOR INITIALIZATION  #
    #######################################

    def _add_tokens(self, tokens: dict[Token, TokenInfo]) -> None:
        """Add tokens to the stored tokens."""
        for token, info in tokens.items():
            if token in self._tokens:
                logging.warning(f"Token <{token}> already exists!")
            self._tokens[token] = info

    def _add_orders(self, orders: list[Order]) -> None:
        """Add orders to the list of orders."""
        assert all(o.is_valid() for o in orders)

        for order in orders:
            if order.order_id in self._orders:
                raise ValueError(f"Order pool_id <{order.order_id}> already exists!")
            self._orders[order.order_id] = order

    def _add_uniswaps(self, uniswaps: list[Uniswap]) -> None:
        """Add pools to the list of uniswap pools."""
        assert all(uni.is_valid() for uni in uniswaps)

        for uni in uniswaps:
            if uni.pool_id in self._uniswaps:
                raise ValueError(f"Uniswap pool_id <{uni.pool_id}> already exists!")
            self._uniswaps[uni.pool_id] = uni

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

    @property
    def ref_token(self) -> Token:
        """Access to ref_token."""
        return self._ref_token

    @ref_token.setter
    def ref_token(self, token: Token) -> None:
        """Update ref_token."""
        self._ref_token = token

    def token_info(self, token: Token) -> TokenInfo:
        """Get the token info for a specific token."""
        assert isinstance(token, Token)

        if token not in self.tokens:
            raise ValueError(f"Token <{token}> not in batch auction!")

        return self._tokens[token]

    def order(self, order_id: str) -> Order:
        """Get the Order with the given pool_id."""
        assert isinstance(order_id, str)

        order = self._orders.get(order_id)
        if order is None:
            raise ValueError(f"Order <{order_id}> not in batch auction!")

        return order

    def uniswap(self, uniswap_id: str) -> Uniswap:
        """Get the Uniswap with the given pool_id."""
        assert isinstance(uniswap_id, str)

        uniswap = self._uniswaps.get(uniswap_id)
        if uniswap is None:
            raise ValueError(f"Uniswap <{uniswap_id}> not in batch auction!")

        return uniswap

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

    def validate(self) -> bool:
        """Check data integrity.

        Returns:
            True, if everything is correct.

        """
        # [1] Check tokens.
        assert hasattr(self, "_tokens")
        assert isinstance(self._tokens, dict)
        assert all(isinstance(t, Token) for t in self._tokens)

        # [2] Check orders.
        assert hasattr(self, "_orders")
        assert isinstance(self._orders, dict)
        assert all(isinstance(oID, str) for oID in self._orders)
        assert all(isinstance(o, Order) and o.is_valid() for o in self._orders.values())
        assert all(oID == o.order_id for oID, o in self._orders.items())

        # [3] Check uniswaps.
        assert hasattr(self, "_uniswaps")
        assert isinstance(self._uniswaps, dict)
        assert all(isinstance(uniID, str) for uniID in self._uniswaps)
        assert all(
            isinstance(uni, Uniswap) and uni.is_valid()
            for uni in self._uniswaps.values()
        )
        assert all(uniID == uni.pool_id for uniID, uni in self._uniswaps.items())

        # [4] Check prices.
        assert hasattr(self, "prices")
        assert isinstance(self.prices, dict) or self.prices is None
        if isinstance(self.prices, dict):
            assert all(t in self.tokens for t in self.prices)
            assert all(
                isinstance(p, (int, float, Decimal)) or p is None
                for p in self.prices.values()
            )

        return True

    #################################
    #  SOLUTION PROCESSING METHODS  #
    #################################

    def evaluate_objective_functions(self) -> Optional[dict]:
        """Compute values of different objective functions.

        Returns:
            A dictionary with several metrics regarding objective.
        """
        if not self.ref_token:
            return None

        # Init objective values.
        # TODO - use a data class for this.
        res = {
            "volume": TokenBalance(0, self.ref_token),
            "surplus": TokenBalance(0, self.ref_token),
            "fees": TokenBalance(0, self.ref_token),
            "cost": TokenBalance(0, self.ref_token),
            "objective_surplus_fees_costs": TokenBalance(0, self.ref_token),
        }

        if not self.has_solution():
            return res

        p_ref_token = self.prices[self.ref_token]

        if p_ref_token is None:
            return None

        def vol_in_ref_token(val: Decimal) -> TokenBalance:
            return TokenBalance(val / p_ref_token, self.ref_token)

        for order in self.orders:
            order_objective = order.evaluate_objective(self.ref_token, self.prices)
            res["volume"] += order_objective.volume
            res["surplus"] += order_objective.surplus
            res["fees"] += order_objective.fees
            res["cost"] += order_objective.cost

        for amm in self.uniswaps:
            if not amm.balance_update1.is_zero() and amm.cost is not None:
                res["cost"] += vol_in_ref_token(
                    amm.cost.as_decimal() * self.prices[amm.cost.token]
                )

        res["objective_surplus_fees_costs"] = res["surplus"] + res["fees"] - res["cost"]
        return res

    def has_solution(self) -> bool:
        """Check if batch auction has a valid solution.

        Returns:
            True, if there are prices + order execution info; False otherwise.

        """
        self.validate()

        # (1) Check clearing prices.
        if not self.prices:
            return False

        # (2) Check order execution.
        for order in self.orders:
            if not order.is_executed():
                logging.warning(
                    f"No execution information for order <{order.order_id}>!"
                )
                return False

        return True

    #######################
    #  AUXILIARY METHODS  #
    #######################

    def get_orderbook(
        self, sort_by_limit_price: bool = False
    ) -> dict[tuple[Token, Token], list[Order]]:
        """Build orderbook-like data structure.

        Kwargs:
            sort_by_limit_price: Sort orders by limit price from best-to-worst, or not.

        """
        orderbook = {}

        orders = self.orders
        if sort_by_limit_price:
            orders.sort(
                key=lambda o: o.get_max_limit().convert_unit(o.sell_token).as_decimal()
            )

        for order in orders:
            sell_token, buy_token = order.sell_token, order.buy_token

            if (sell_token, buy_token) not in orderbook:
                orderbook[(sell_token, buy_token)] = [order]
            else:
                orderbook[sell_token, buy_token].append(order)

        return orderbook

    def get_orders(
        self, tokenpairs: Optional[list[tuple]] = None, tokens: Optional[list] = None
    ) -> list[Order]:
        """Get all orders from the orderbook.
        Args:
            tokenpairs: list of (directed) token pairs for which to get orders.
            tokens: list of tokens for which to get all orders.

        Returns:
            A list of orders, sorted by their IDs.
        """
        orders = []

        # Get orders on token pairs.
        if tokenpairs is not None:
            for pair in tokenpairs:
                if pair not in self.get_tokenpairs():
                    logging.warning(f"Token pair {pair} not present in current batch!")

            for order in self.orders:
                pair = (order.sell_token, order.buy_token)
                rev_pair = (pair[1], pair[0])
                if pair in tokenpairs or rev_pair in tokenpairs:
                    orders.append(order)

        # Get orders that involve given token.
        elif tokens is not None:

            for token in tokens:
                if token not in self.tokens:
                    logging.warning(
                        f"Token {token} not present in current batch auction!"
                    )

            for order in self.orders:
                if any(t in tokens for t in [order.buy_token, order.sell_token]):
                    orders.append(order)

        # Get all orders.
        else:
            orders = self.orders

        return sorted(orders, key=lambda o: o.order_id)

    def get_uniswaps_by_tokenpair(
        self, tokenpair: tuple[Token, Token]
    ) -> list[Uniswap]:
        """Get a list of AMMs acting on the given token pair.

        Args:
            tokenpair: Token pair for which to get uniswaps.

        Returns:
            A list of uniswaps.

        """
        return [u for u in self.uniswaps if u.tokens == set(tokenpair)]

    def get_tokenpairs(self) -> list[tuple[Token, Token]]:
        """Get all pairs of tokens present.
        Returns:
            An iterator of tuples.
        """
        return list(itertools.combinations(self.tokens, 2))

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
        return str(self)


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


def load_orders(orders_serialized: OrdersSerializedType) -> list[Order]:
    """Load dict of orders as order pool_id -> order data.

    Args:
        orders_serialized: dict of order pool_id -> order data dict.

    Returns:
        A list of Order objects.

    """
    if not isinstance(orders_serialized, dict):
        raise ValueError(
            f"The 'orders' field must be a dict, not {type(orders_serialized)}!"
        )

    return [
        Order.from_dict(order_id, order_data)
        for order_id, order_data in orders_serialized.items()
    ]


def load_amms(amms_serialized: UniswapsSerializedType) -> list[Uniswap]:
    """Load list of AMMs.

    NOTE: Currently, the code only supports Uniswap-style AMMs, i.e.,
    constant-product pools with two tokens and equal weights.

    Args:
        amms_serialized: dict of pool_id -> AMM.

    Returns:
        A list of Uniswap objects.

    """
    if not isinstance(amms_serialized, dict):
        raise ValueError(
            f"The 'amms' field must be a dict, not {type(amms_serialized)}!"
        )

    amms = []

    for amm_id, amm_data in amms_serialized.items():
        amm = Uniswap.from_dict(amm_id, amm_data)
        if amm is not None:
            amms.append(amm)

    return sorted(amms, key=lambda a: a.pool_id)


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
                k = inflection.underscore(k)
                try:
                    # Convert to str first to avoid float-precision artifacts:
                    # Decimal(0.1)   -> Decimal('0.10000000000000000555...')
                    # Decimal('0.1') -> Decimal(0.1)
                    val = Decimal(str(val))
                except decimal.InvalidOperation:
                    pass
                token_info[k] = val
        tokens_dict[token] = TokenInfo(token, **token_info)

    return tokens_dict


def select_ref_token(tokens: dict[Token, TokenInfo]) -> Token:
    """Select ref_token from the list of tokens."""
    return select_token_with_highest_normalize_priority(tokens)

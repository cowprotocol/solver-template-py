"""Representation of a limit order."""
from __future__ import annotations

import json
import logging
from decimal import Decimal
from enum import Enum
from typing import Optional, Any, Union


from src.models.exchange_rate import ExchangeRate as XRate
from src.models.token import Token, TokenBalance
from src.models.types import NumericType
from src.util.constants import Constants
from src.util.numbers import decimal_to_str

OrderSerializedType = dict[str, Any]
OrdersSerializedType = dict[str, OrderSerializedType]


class OrderMatchType(Enum):
    """Enum for different Order Matching"""

    LHS_FILLED = "LhsFilled"
    RHS_FILLED = "RhsFilled"
    BOTH_FILLED = "BothFilled"


# TODO - use dataclass for this.
class Order:
    """Representation of a limit order.
    An order is specified with 3 bounds:
        * maximum amount of buy-token to be bought
        * maximum amount of sell-token to be sold
        * limit exchange rate of buy- vs. sell-token.

    Depending on which of the bounds are set,
    the order represents a classical limit {buy|sell} order,
    a cost-bounded {buy|sell} order or a {buy|sell} market order.
    """

    def __init__(
        self,
        order_id: str,
        buy_token: Token,
        sell_token: Token,
        buy_amount: Decimal,
        sell_amount: Decimal,
        is_sell_order: bool,
        allow_partial_fill: bool = False,
        is_liquidity_order: bool = False,
        has_atomic_execution: bool = False,
        fee: Optional[TokenBalance] = None,
        cost: Optional[TokenBalance] = None,
    ) -> None:
        """Initialize.

        Args:
            order_id: Order pool_id.
            buy_token: Token to be bought.
            sell_token: Token to be sold.

        Kwargs:
            max_buy_amount: Maximum amount of buy-token to be bought, or None.
            max_sell_amount: Maximum amount of sell-token to be sold, or None.
            max_limit: Limit exchange rate for order.
            allow_partial_fill: Can order be partially matched, or not.
            is_liquidity_order: Is the order from a market maker, or not.
            has_atomic_execution: Needs to executed atomically, or not.
            fee: Fee contribution of the order to the objective.
            cost: Cost of including the order in the solution.
            exec_buy_amount: Matched amount of buy-token in solution.
            exec_sell_amount: Matched amount of sell-token in solution.
        """
        if buy_token == sell_token:
            raise ValueError("sell- and buy-token cannot be equal!")

        if not (buy_amount > 0 and sell_amount > 0):
            raise ValueError(
                f"buy {buy_amount} and sell {sell_amount} amounts must be positive!"
            )

        self.order_id = order_id
        self.buy_token = buy_token
        self.sell_token = sell_token
        self.buy_amount = buy_amount
        self.sell_amount = sell_amount
        self.is_sell_order = is_sell_order
        self.allow_partial_fill = allow_partial_fill
        self.is_liquidity_order = is_liquidity_order
        self.has_atomic_execution = has_atomic_execution
        self.fee: Optional[TokenBalance] = fee
        self.cost: Optional[TokenBalance] = cost

        # Stuff that isn't part of the constructor parameters.
        self.exec_buy_amount: Optional[TokenBalance] = None
        self.exec_sell_amount: Optional[TokenBalance] = None

    @classmethod
    def from_dict(cls, order_id: str, data: OrderSerializedType) -> Order:
        """
        Read Order object from order data dict.
        Args:
            order_id: ID of order
            data: Dict of order data.
        """

        required_attributes = [
            "sell_token",
            "buy_token",
            "sell_amount",
            "buy_amount",
            "is_sell_order",
            "allow_partial_fill",
        ]

        for attr in required_attributes:
            if attr not in data:
                raise ValueError(f"Missing field '{attr}' in order <{order_id}>!")

        return Order(
            order_id=order_id,
            buy_token=Token(data["buy_token"]),
            sell_token=Token(data["sell_token"]),
            buy_amount=Decimal(data["buy_amount"]),
            sell_amount=Decimal(data["sell_amount"]),
            is_sell_order=bool(data["is_sell_order"]),
            allow_partial_fill=bool(data["allow_partial_fill"]),
            fee=TokenBalance.parse(data.get("fee"), allow_none=True),
            cost=TokenBalance.parse(data.get("cost"), allow_none=True),
        )

    def as_dict(self) -> OrderSerializedType:
        """Return Order object as dictionary."""
        # Currently, only limit buy or sell orders be handled.
        order_dict = {
            "sell_token": str(self.sell_token),
            "buy_token": str(self.buy_token),
            "sell_amount": decimal_to_str(self.sell_amount),
            "buy_amount": decimal_to_str(self.buy_amount),
            "allow_partial_fill": self.allow_partial_fill,
            "is_sell_order": self.is_sell_order,
            "exec_sell_amount": decimal_to_str(self.exec_sell_amount.as_decimal())
            if self.exec_sell_amount is not None
            else "0",
            "exec_buy_amount": decimal_to_str(self.exec_buy_amount.as_decimal())
            if self.exec_buy_amount is not None
            else "0",
        }

        if self.fee is not None:
            order_dict["fee"] = {
                "token": str(self.fee.token),
                "amount": decimal_to_str(self.fee.as_decimal()),
            }

        if self.cost is not None:
            order_dict["cost"] = {
                "token": str(self.cost.token),
                "amount": decimal_to_str(self.cost.as_decimal()),
            }

        return order_dict

    @property
    def max_limit(self) -> XRate:
        """Max limit of the order as an exchange rate"""
        return XRate(
            tb1=TokenBalance(self.sell_amount, self.sell_token),
            tb2=TokenBalance(self.buy_amount, self.buy_token),
        )

    @property
    def max_buy_amount(self) -> Optional[TokenBalance]:
        """None for sell-orders"""
        if not self.is_sell_order:
            return TokenBalance.parse_amount(self.buy_amount, self.buy_token)
        return None

    @property
    def max_sell_amount(self) -> Optional[TokenBalance]:
        """None for buy-orders"""
        if self.is_sell_order:
            return TokenBalance.parse_amount(self.sell_amount, self.sell_token)
        return None

    @property
    def tokens(self) -> set[Token]:
        """Return the buy and sell tokens."""
        return {self.buy_token, self.sell_token}

    #####################
    #  UTILITY METHODS  #`
    #####################

    def overlaps(self, other: Order) -> bool:
        """
        Determine if one order can be matched with another.
        opposite {buy|sell} tokens and matching prices
        """
        token_conditions = [
            self.buy_token == other.sell_token,
            self.sell_token == other.buy_token,
        ]
        if not all(token_conditions):
            return False

        return (
            self.buy_amount * other.buy_amount <= other.sell_amount * self.sell_amount
        )

    def match_type(self, other: Order) -> Optional[OrderMatchType]:
        """Determine to what extent two orders match"""
        if not self.overlaps(other):
            return None

        if self.buy_amount < other.sell_amount and self.sell_amount < other.buy_amount:
            return OrderMatchType.LHS_FILLED

        if self.buy_amount > other.sell_amount and self.sell_amount > other.buy_amount:
            return OrderMatchType.RHS_FILLED

        return OrderMatchType.BOTH_FILLED

    def is_executable(self, xrate: XRate, xrate_tol: Decimal = Decimal("1e-6")) -> bool:
        """Determine if the order limit price satisfies a given market rate.

        Args:
            xrate: Exchange rate.
            xrate_tol: Accepted violation of the limit exchange rate constraint
                       per unit of buy token (default: 1e-6).
        Returns:
            True, if order can be executed; False otherwise.
        """
        buy_token, sell_token = self.buy_token, self.sell_token
        if xrate.tokens != {buy_token, sell_token}:
            raise ValueError(
                f"Exchange rate and order tokens do not "
                f"match: {xrate} vs. <{buy_token}> | <{sell_token}>!"
            )

        assert xrate_tol >= 0
        converted_buy = xrate.convert_unit(buy_token)
        converted_sell = self.max_limit.convert_unit(buy_token)
        return bool(converted_buy <= (converted_sell * (1 + xrate_tol)))

    def execute(
        self,
        buy_amount_value: NumericType,
        sell_amount_value: NumericType,
        buy_token_price: Union[float, Decimal] = 0,
        sell_token_price: Union[float, Decimal] = 0,
        amount_tol: Decimal = Decimal("1e-8"),
        xrate_tol: Decimal = Decimal("1e-6"),
    ) -> None:
        """Execute the order at given amounts.

        Args:
            buy_amount_value: Buy amount.
            sell_amount_value: Sell amount.
            buy_token_price: Buy-token price.
            sell_token_price: Sell-token price.
            amount_tol: Accepted violation of the limit buy/sell amount constraints.
            xrate_tol: Accepted violation of the limit exchange rate constraint
                       per unit of buy token (default: 1e-6).
        """
        assert buy_amount_value >= -amount_tol
        assert sell_amount_value >= -amount_tol
        assert buy_token_price >= 0
        assert sell_token_price >= 0

        buy_token, sell_token = self.buy_token, self.sell_token

        buy_amount = TokenBalance(buy_amount_value, buy_token)
        sell_amount = TokenBalance(sell_amount_value, sell_token)

        xmax = self.max_buy_amount
        ymax = self.max_sell_amount

        # (a) Check buyAmount: if too much above maxBuyAmount --> error!
        if xmax is not None:
            if buy_amount > xmax * (
                1 + amount_tol
            ) and buy_amount > xmax + TokenBalance(amount_tol, buy_token):
                raise ValueError(
                    f"Invalid execution request for "
                    f"order <{self.order_id}>: "
                    f"buy amount (exec) : {buy_amount.balance} "
                    f"buy amount (max)  : {xmax.balance}"
                )

            buy_amount = min(buy_amount, xmax)

        # (b) Check sellAmount: if too much above maxSellAmount --> error!
        if ymax is not None:
            if sell_amount > ymax * (
                1 + amount_tol
            ) and sell_amount > ymax + TokenBalance(amount_tol, sell_token):
                message = (
                    f"Invalid execution request for "
                    f"order <{self.order_id}>: "
                    f"sell (exec) : {sell_amount.balance} "
                    f"sell (max)  : {ymax.balance}"
                )
                logging.error(message)
                if Constants.RAISE_ON_MAX_SELL_AMOUNT_VIOLATION:
                    raise ValueError(message)
            sell_amount = min(sell_amount, ymax)

        # (c) if any amount is very small, set to zero.
        if any(
            [
                buy_amount <= TokenBalance(amount_tol, buy_token),
                sell_amount <= TokenBalance(amount_tol, sell_token),
            ]
        ):
            buy_amount = TokenBalance(0.0, buy_token)
            sell_amount = TokenBalance(0.0, sell_token)

        # Check limit price.
        if buy_amount > 0:
            assert sell_amount > 0
            xrate = XRate(buy_amount, sell_amount)
            if not self.is_executable(xrate, xrate_tol=xrate_tol):
                message = (
                    f"Invalid execution request for order <{self.order_id}>: "
                    f"buy amount (exec): {buy_amount.balance} "
                    f"sell amount (exec): {sell_amount.balance} "
                    f"xrate (exec): {xrate} "
                    f"limit (max): {self.max_limit}"
                )
                logging.error(message)
                if Constants.RAISE_ON_LIMIT_XRATE_VIOLATION:
                    raise ValueError(message)

        # Store execution information.
        self.exec_buy_amount = buy_amount
        self.exec_sell_amount = sell_amount

    def is_executed(self) -> bool:
        """Check if order has already been executed."""
        return self.exec_buy_amount is not None and self.exec_sell_amount is not None

    def __str__(self) -> str:
        """Represent as string."""
        return json.dumps(self.as_dict(), indent=2)

    def __repr__(self) -> str:
        """Represent as short string."""
        return f"Order: {self.order_id}"

    def __hash__(self) -> int:
        return hash(self.order_id)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Order):
            return NotImplemented
        if self.order_id != other.order_id:
            return False
        assert vars(self) == vars(other)
        return True

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, Order):
            return NotImplemented

        return self.order_id < other.order_id

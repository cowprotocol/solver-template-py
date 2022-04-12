"""Representation of a limit order."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from decimal import Decimal
from typing import Optional, Any

import inflection

from src.models.exchange_rate import ExchangeRate as XRate
from src.models.token import Token, TokenBalance, Amount
from src.models.types import NumericType
from src.util.constants import Constants
from src.util.numbers import decimal_to_str

OrderSerializedType = dict[str, Any]
OrdersSerializedType = dict[str, OrderSerializedType]


@dataclass
class ObjectiveValue:
    """Simplifies return type of Order.evaluate_objective"""

    volume: TokenBalance
    surplus: TokenBalance
    fees: TokenBalance
    cost: TokenBalance

    @classmethod
    def default(cls, ref_token: Token) -> ObjectiveValue:
        """Default Objective value of zero in all fields for given ref_token"""
        return cls(
            volume=TokenBalance.default(ref_token),
            surplus=TokenBalance.default(ref_token),
            fees=TokenBalance.default(ref_token),
            cost=TokenBalance.default(ref_token),
        )


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
        max_limit: XRate,
        max_buy_amount: Optional[Amount] = None,
        max_sell_amount: Optional[Amount] = None,
        allow_partial_fill: bool = False,
        is_liquidity_order: bool = False,
        has_atomic_execution: bool = False,
        fee: Optional[TokenBalance] = None,
        cost: Optional[TokenBalance] = None,
        exec_buy_amount: Optional[Amount] = None,
        exec_sell_amount: Optional[Amount] = None,
        **kwargs: dict[str, Any],
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
        # Consistency checks.
        assert all(isinstance(t, Token) for t in [buy_token, sell_token])
        if buy_token == sell_token:
            raise ValueError("Sell- and buy-token cannot be equal!")

        self.order_id = order_id

        # Init order data.
        self._buy_token = buy_token
        self._sell_token = sell_token

        # Store fill-or-kill property.
        self.allow_partial_fill = allow_partial_fill

        # Store liquidity property.
        self.is_liquidity_order = is_liquidity_order

        # Store atomic execution property.
        self.has_atomic_execution = has_atomic_execution

        self._max_buy_amount = TokenBalance.parse_amount(max_buy_amount, buy_token)
        self._max_sell_amount = TokenBalance.parse_amount(max_sell_amount, sell_token)
        self._max_limit = self._parse_limit(max_limit)

        self._fee = fee
        self._cost = cost

        self._exec_buy_amount = TokenBalance.parse_amount(exec_buy_amount, buy_token)
        self._exec_sell_amount = TokenBalance.parse_amount(exec_sell_amount, sell_token)

        self.exec_rate: Optional[XRate] = None
        self.exec_volume: Optional[Decimal] = None

        # Store additional keyword-arguments.
        for k, val in kwargs.items():
            setattr(self, k, val)

    @classmethod
    def from_dict(cls, order_id: str, order_data: OrderSerializedType) -> Order:
        """Read Order object from order data dict.

        Note: Currently, the smart contract format only supports limit sell orders.

        Args:
            order_id: pool_id of order
            order_data: Dict of order data.

        Kwargs:
            max_len_order_id: Maximum length of order pool_id (for leading zeroes).

        Returns:
            An Order object.

        """
        if not isinstance(order_id, str):
            raise ValueError(f"Order pool_id must be a string, not <{type(order_id)}>!")

        if not isinstance(order_data, dict):
            raise ValueError(f"Order data must be a dict, not <{type(order_data)}>!")

        attr_mandatory = [
            "sell_token",
            "buy_token",
            "sell_amount",
            "buy_amount",
            "is_sell_order",
            "allow_partial_fill",
        ]

        for attr in attr_mandatory:
            if attr not in order_data:
                raise ValueError(f"Missing field '{attr}' in order <{order_id}>!")

        buy_token = Token(order_data["buy_token"])
        sell_token = Token(order_data["sell_token"])

        buy_amount = Decimal(order_data["buy_amount"])
        sell_amount = Decimal(order_data["sell_amount"])

        if not (buy_amount > 0 and sell_amount > 0):
            raise ValueError(
                f"Order buy {buy_amount} and sell {sell_amount} amounts must be positive!"
            )

        limit_xrate = XRate(
            TokenBalance(sell_amount, sell_token), TokenBalance(buy_amount, buy_token)
        )

        allow_partial_fill = bool(order_data["allow_partial_fill"])

        is_sell_order = bool(order_data["is_sell_order"])
        if is_sell_order:
            max_buy_amount = None
            max_sell_amount = sell_amount
        else:
            max_buy_amount = buy_amount
            max_sell_amount = None

        kwargs = {
            inflection.underscore(k): v
            for k, v in order_data.items()
            if k not in attr_mandatory
        }
        kwargs["fee"] = TokenBalance.parse(
            order_data.get("fee"), allow_negative=False, allow_none=True
        )
        kwargs["cost"] = TokenBalance.parse(
            order_data.get("cost"), allow_negative=False, allow_none=True
        )
        kwargs["exec_buy_amount"] = TokenBalance.parse(
            order_data.get("exec_buy_amount"), allow_none=True
        )
        kwargs["exec_sell_amount"] = TokenBalance.parse(
            order_data.get("exec_sell_amount"), allow_none=True
        )

        return Order(
            order_id,
            buy_token,
            sell_token,
            max_buy_amount=max_buy_amount,
            max_sell_amount=max_sell_amount,
            max_limit=limit_xrate,
            allow_partial_fill=allow_partial_fill,
            **kwargs,
        )

    def as_dict(self) -> OrderSerializedType:
        """Return Order object as dictionary."""
        # Currently, only limit buy or sell orders be handled.

        if self.max_buy_amount is None:
            sell_amount = (
                self.max_sell_amount.as_decimal()
                if self.max_sell_amount
                else Decimal(0)
            )
            buy_amount = self.max_limit.convert(self.max_sell_amount).as_decimal()
            is_sell_order = True
        else:
            buy_amount = (
                self.max_buy_amount.as_decimal() if self.max_buy_amount else Decimal(0)
            )
            sell_amount = self.max_limit.convert(self.max_buy_amount).as_decimal()
            is_sell_order = False

        order_dict = {
            "sell_token": str(self.sell_token),
            "buy_token": str(self.buy_token),
            "sell_amount": decimal_to_str(sell_amount),
            "buy_amount": decimal_to_str(buy_amount),
            "allow_partial_fill": self.allow_partial_fill,
            "is_sell_order": is_sell_order,
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

        if self.exec_sell_amount is not None:
            order_dict["exec_sell_amount"] = decimal_to_str(
                self.exec_sell_amount.as_decimal()
            )

        if self.exec_buy_amount is not None:
            order_dict["exec_buy_amount"] = decimal_to_str(
                self.exec_buy_amount.as_decimal()
            )

        return order_dict

    ############################
    #  AUXILIARY INIT METHODS  #
    ############################

    def _parse_limit(self, xrate: XRate) -> Decimal:
        """Auxiliary method to get limit price from parameter xrate.

        Args:
            xrate: Limit exchange rate.

        Returns:
            A float value x satisfying x[sell_token] = 1[buy_token], or None

        """
        buy_token, sell_token = self.buy_token, self.sell_token
        if not xrate.tokens == {buy_token, sell_token}:
            logging.error(
                f"Exchange rate and order tokens do not match: "
                f"{xrate} vs. <{buy_token}> | <{sell_token}>!"
            )
            raise ValueError
        return xrate.convert_unit(buy_token).as_decimal()

    ####################
    #  ACCESS METHODS  #
    ####################

    @property
    def buy_token(self) -> Token:
        """Return the buy-token."""
        return self._buy_token

    @property
    def sell_token(self) -> Token:
        """Return the sell-token."""
        return self._sell_token

    @property
    def tokens(self) -> set[Token]:
        """Return the buy and sell tokens."""
        return {self.buy_token, self.sell_token}

    @property
    def max_buy_amount(self) -> Optional[TokenBalance]:
        """Return the maximum buy amount."""
        return self._max_buy_amount

    @property
    def max_sell_amount(self) -> Optional[TokenBalance]:
        """Return the maximum sell amount."""
        return self._max_sell_amount

    def get_max_limit(self) -> XRate:
        """Return the max_limit as XRate object."""

        return XRate(
            TokenBalance(1.0, self.buy_token),
            TokenBalance(self._max_limit, self.sell_token),
        )

    def set_max_limit(self, xrate: XRate) -> None:
        """Set the max_limit as XRate object."""
        self._max_limit = self._parse_limit(xrate)

    max_limit = property(get_max_limit, set_max_limit)

    @property
    def is_sell_order(self) -> bool:
        """true if order is sell order"""
        return self.max_sell_amount is not None

    @property
    def is_buy_order(self) -> bool:
        """true if order is buy order"""
        return self.max_buy_amount is not None

    @property
    def fee(self) -> Optional[TokenBalance]:
        """Return the fee contribution."""
        return self._fee

    @property
    def cost(self) -> Optional[TokenBalance]:
        """Return the execution cost."""
        return self._cost

    @property
    def exec_buy_amount(self) -> TokenBalance:
        """Return the executed buy amount."""
        return self._exec_buy_amount

    @exec_buy_amount.setter
    def exec_buy_amount(self, value: Optional[Amount]) -> None:
        self._exec_buy_amount = TokenBalance.parse_amount(value, self.buy_token)

    @property
    def exec_sell_amount(self) -> TokenBalance:
        """Return the executed sell amount."""
        return self._exec_sell_amount

    @exec_sell_amount.setter
    def exec_sell_amount(self, value: Optional[Amount]) -> None:
        self._exec_sell_amount = TokenBalance.parse_amount(value, self.sell_token)

    #####################
    #  UTILITY METHODS  #
    #####################

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

        # Buy @ AT MOST limit rate.
        max_limit = self.max_limit
        if max_limit is None:  # market order.
            return True

        assert xrate_tol >= 0
        converted_buy = xrate.convert_unit(buy_token)
        converted_sell = max_limit.convert_unit(sell_token)
        return bool(converted_buy <= (converted_sell * (1 + xrate_tol)))

    def execute(
        self,
        buy_amount_value: NumericType,
        sell_amount_value: NumericType,
        buy_token_price: float | Decimal = 0,
        sell_token_price: float | Decimal = 0,
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
        # We don't check types because type mypy check that.
        assert buy_amount_value >= -amount_tol
        assert sell_amount_value >= -amount_tol
        assert buy_token_price is None or buy_token_price >= 0
        assert sell_token_price is None or sell_token_price >= 0

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
        if buy_amount <= TokenBalance(
            amount_tol, buy_token
        ) or sell_amount <= TokenBalance(amount_tol, sell_token):
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

        if buy_token_price > 0 and sell_token_price > 0:
            self.exec_rate = XRate.from_prices(
                (buy_token, buy_token_price), (sell_token, sell_token_price)
            )
            self.exec_volume = (buy_amount * buy_token_price).as_decimal()

    def is_executed(self) -> bool:
        """Check if order has already been executed."""
        assert self.is_valid()
        return all(v is not None for v in [self.exec_buy_amount, self.exec_sell_amount])

    def is_valid(self) -> bool:
        """Validate the order against a list of tokens."""

        buy_exec_valid = validate_execution(
            self.exec_buy_amount, self.buy_token, self.max_buy_amount
        )
        sell_exec_valid = validate_execution(
            self.exec_sell_amount, self.sell_token, self.max_sell_amount
        )

        if hasattr(self, "exec_rate") and self.exec_rate is not None:
            if not isinstance(self.exec_rate, XRate):
                return False
            if self.exec_rate.tokens != self.max_limit.tokens:
                return False

        return bool(buy_exec_valid and sell_exec_valid)

    def evaluate_objective(
        self,
        ref_token: Token,
        prices: dict,
    ) -> ObjectiveValue:
        """
        Evaluates the objective values for a single order
        given execution prices in reference token
        """
        result = ObjectiveValue.default(ref_token)

        def vol_in_ref_token(val: Decimal) -> TokenBalance:
            return TokenBalance(val / prices[ref_token], ref_token)

        buy_token, sell_token = self.buy_token, self.sell_token
        buy_price, sell_price = prices.get(buy_token), prices.get(sell_token)

        if buy_price is None or sell_price is None:
            return result

        exec_buy = self.exec_buy_amount.as_decimal()
        exec_sell = self.exec_sell_amount.as_decimal()

        result.volume += vol_in_ref_token(exec_sell * sell_price)

        if self.max_buy_amount is not None and self.max_sell_amount is not None:
            # Double-sided orders.
            logging.warning(
                f"surplus is not defined for double-sided order <{self.order_id}>!"
            )

        elif self.max_limit is None:
            # Market orders.
            logging.warning(
                f"Welfare is not defined for market order <{self.order_id}>!"
            )

        elif not self.is_liquidity_order and self.max_buy_amount is not None:
            # Limit buy orders.
            lim = self.max_limit.convert_unit(buy_token).as_decimal()
            result.surplus += vol_in_ref_token(
                exec_buy * (lim * sell_price - buy_price)
            )

        elif not self.is_liquidity_order and self.max_sell_amount is not None:
            # Limit sell orders.
            lim = self.max_limit.convert_unit(sell_token).as_decimal()
            result.surplus += vol_in_ref_token(
                exec_sell * (sell_price - lim * buy_price)
            )
        else:
            assert self.is_liquidity_order

        if exec_buy > 0:
            if self.fee is not None:
                if self.is_sell_order:
                    assert isinstance(self.max_sell_amount, TokenBalance)
                    max_vol = self.max_sell_amount.as_decimal() * sell_price
                else:
                    assert isinstance(self.max_buy_amount, TokenBalance)
                    max_vol = self.max_buy_amount.as_decimal() * buy_price

                exec_vol = exec_buy * buy_price

                fill_ratio = exec_vol / max_vol

                result.fees += vol_in_ref_token(
                    fill_ratio * self.fee.as_decimal() * prices[self.fee.token]
                )

            if self.cost is not None:
                result.cost += vol_in_ref_token(
                    self.cost.as_decimal() * prices[self.cost.token]
                )
        return result

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


def serialize_orders(orders: list[Order]) -> OrdersSerializedType:
    """Return dict of order pool_id -> order details."""
    return {o.order_id: o.as_dict() for o in orders}


def validate_execution(
    exec_amt: Optional[TokenBalance], token: Token, max_amt: Optional[TokenBalance]
) -> bool:
    """Similar method used to validate both executed buy and sell amounts for order"""
    assert isinstance(exec_amt, TokenBalance)
    assert isinstance(max_amt, TokenBalance)

    validity_conditions = [
        exec_amt.token == token,
        exec_amt.balance >= 0,
        max_amt and exec_amt <= max_amt,
    ]
    return exec_amt is None or all(validity_conditions)

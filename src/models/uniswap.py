"""Representation of Uniswap pool."""
from __future__ import annotations

import json
import logging
from decimal import Decimal
from typing import Optional, Any

from src.models.exchange_rate import ExchangeRate as XRate
from src.models.token import Token, TokenBalance, Amount
from src.models.types import NumericType
from src.util.enums import AMMKind
from src.util.exec_plan_coords import ExecPlanCoords
from src.util.numbers import decimal_to_str

FeeType = float | Decimal
UniswapSerializedType = dict[str, Any]
UniswapsSerializedType = dict[str, UniswapSerializedType]


class Uniswap:
    """Representation of a Uniswap pool.

    An Uniswap pool is represented by two token balances.
    """

    def __init__(
        self,
        pool_id: str,
        balance1: TokenBalance,
        balance2: TokenBalance,
        fee: FeeType,
        cost: Optional[TokenBalance] = None,
        mandatory: bool = False,
        kind: AMMKind = AMMKind.UNISWAP,
        **kwargs: dict[str, Any],
    ):
        """Initialize.

        Args:
            pool_id: Uniswap pool pool_id.
            balance1: TokenBalance of first token.
            balance2: TokenBalance of second token.
            fee: Uniswap fee percentage.

        Kwargs:
            cost: Cost of using the Uniswap pool.
            mandatory: Is pool usage mandatory when price moves, or not.

        """
        # Store given pool pool_id.
        self.pool_id = pool_id

        # Init pool balances.
        self._balance1 = balance1
        self._balance2 = balance2

        self._fee = fee if isinstance(fee, Decimal) else Decimal(fee)
        self._cost = cost

        self.mandatory = mandatory

        self.kind = kind

        self._balance_update1 = TokenBalance.default(balance1.token)
        self._balance_update2 = TokenBalance.default(balance1.token)

        self._exec_plan_coords = ExecPlanCoords(None, None)

        # Consistency checks.
        if balance1.token == balance2.token:
            logging.error("Pool tokens cannot be equal!")
            raise ValueError

        if not all(tb.is_positive() for tb in [balance1, balance2]):
            message = (
                f"Uniswap <{self.pool_id}>: balance1={balance1} balance2={balance2}"
            )
            logging.error(message)
            raise ValueError("Both token balances must be positive!")

        # Store additional keyword-arguments.
        for k, val in kwargs.items():
            setattr(self, k, val)

    @classmethod
    def from_dict(
        cls, amm_id: str, amm_data: UniswapSerializedType
    ) -> Optional[Uniswap]:
        """Read amm object from data dict.

        NOTE: Currently, the code only supports Uniswap-style AMMs, i.e.,
        constant-product pools with two tokens and equal weights.

        Args:
            amm_id: AMM pool_id.
            amm_data: Dict of uniswap data.

        Returns:
            A Uniswap object.

        """
        # Validate Input
        if not isinstance(amm_id, str):
            raise ValueError(f"AMM pool_id must be a string, not {type(amm_id)}!")

        if not isinstance(amm_data, dict):
            raise ValueError(f"AMM data must be a dict, not {type(amm_data)}!")

        attr_mandatory = ["kind", "reserves", "fee"]

        for attr in attr_mandatory:
            if attr not in amm_data:
                raise ValueError(f"Missing field '{attr}' in amm <{amm_id}>!")

        kind = AMMKind(amm_data["kind"])
        reserves = amm_data.get("reserves")
        input_weight = None

        if kind == AMMKind.CONSTANT_PRODUCT:
            # Parse UniswapV2/Sushiswap pools.
            if not isinstance(reserves, dict):
                raise ValueError(
                    f"AMM <{amm_id}>: 'reserves' must be a dict of Token -> amount!"
                )
            if len(reserves) != 2:
                message = f"AMM <{amm_id}>: ConstantProduct AMMs are only supported with 2 tokens!"
                logging.warning(message)
                return None
            balance1, balance2 = [
                TokenBalance(Decimal(b), Token(t)) for t, b in reserves.items()
            ]

        elif kind == AMMKind.WEIGHTED_PRODUCT:
            # Parse Balancer weighted constant-product pools.
            if not (
                isinstance(reserves, dict)
                and all(
                    isinstance(reserve_info, dict) and key in reserve_info
                    for reserve_info in reserves.values()
                    for key in ["balance", "weight"]
                )
            ):
                raise ValueError(
                    f"AMM <{amm_id}>: 'reserves' must be a dict "
                    f"of Token -> {'balance': .., 'weight': ..}"
                )
            if (
                len(reserves) != 2
                or len(set(b["weight"] for b in reserves.values())) > 1
            ):
                logging.warning(
                    f"AMM <{amm_id}>: WeightedProduct AMMs are only supported "
                    "with 2 tokens and equal weights!"
                )
                return None
            balance1, balance2 = [
                TokenBalance(
                    Decimal(b["balance"]),
                    Token(t),
                )
                for t, b in reserves.items()
            ]
            input_weight = list(reserves.values())[0]["weight"]

        else:
            logging.warning(
                f"AMM <{amm_id}>: type <{kind}> is currently not supported!"
            )
            return None

        fee = Decimal(amm_data["fee"])

        kwargs = {
            "cost": TokenBalance.parse(amm_data.get("cost"), allow_none=True),
            "kind": kind,
        }
        if input_weight:
            kwargs["input_weight"] = input_weight

        return Uniswap(amm_id, balance1, balance2, fee, **kwargs)

    def as_dict(self) -> dict:
        """Return AMM object as dictionary.

        NOTE: Currently, the code only supports Uniswap-style AMMs, i.e.,
        constant-product pools with two tokens and equal weights.

        """
        token1 = str(self.balance1.token)
        token2 = str(self.balance2.token)
        balance1 = decimal_to_str(self.balance1.as_decimal())
        balance2 = decimal_to_str(self.balance2.as_decimal())

        reserves: str | dict
        if self.kind == AMMKind.WEIGHTED_PRODUCT:
            weight = getattr(self, "input_weight", "1")
            reserves = {
                token1: {"balance": balance1, "weight": weight},
                token2: {"balance": balance2, "weight": weight},
            }
        else:
            reserves = {token1: balance1, token2: balance2}

        cost = None
        if self.cost is not None:
            cost = {
                "token": str(self.cost.token),
                "amount": decimal_to_str(self.cost.as_decimal()),
            }

        execution = {}
        if self.is_executed():
            b1_update = self.balance_update1.as_decimal()
            b2_update = self.balance_update2.as_decimal()
            # One update has to be positive and the other negative.
            assert (
                b1_update * b2_update < 0
            ), f"Failed assertion, {b1_update} * {b2_update} < 0"

            # Determine buy- and sell-tokens and -amounts.
            if b1_update > 0:
                buy_token = self.token1
                sell_token = self.token2
                exec_buy_amount = b1_update
                exec_sell_amount = -b2_update

            else:
                buy_token = self.token2
                sell_token = self.token1
                exec_buy_amount = b2_update
                exec_sell_amount = -b1_update

            sequence, position = self.exec_plan_coords.as_tuple()
            if sequence is None or position is None:
                logging.warning(
                    f"AMM <{self.pool_id}>: has balance updates with invalid execution plan"
                )
                exec_plan = None
            else:
                exec_plan = {
                    "sequence": str(sequence),
                    "position": str(position),
                }

            execution = {
                "sell_token": str(sell_token),
                "buy_token": str(buy_token),
                "exec_sell_amount": decimal_to_str(exec_sell_amount),
                "exec_buy_amount": decimal_to_str(exec_buy_amount),
                "exec_plan": exec_plan,
            }

        return {
            "kind": str(self.kind),
            "reserves": reserves,
            "cost": cost,
            "fee": decimal_to_str(self.fee),
            "execution": execution,
        }

    ####################
    #  ACCESS METHODS  #
    ####################

    @property
    def balance1(self) -> TokenBalance:
        """Return the first TokenBalance."""
        return self._balance1

    @balance1.setter
    def balance1(self, value: TokenBalance) -> None:
        self._balance1 = value

    @property
    def balance2(self) -> TokenBalance:
        """Return the second TokenBalance."""
        return self._balance2

    @balance2.setter
    def balance2(self, value: TokenBalance) -> None:
        self._balance2 = value

    @property
    def token1(self) -> Token:
        """Returns token1"""
        return self._balance1.token

    @property
    def token2(self) -> Token:
        """Returns token2"""
        return self._balance2.token

    @property
    def fee(self) -> Decimal:
        """Return the fee percentage."""
        return self._fee

    @fee.setter
    def fee(self, value: FeeType) -> None:
        self._fee = Decimal(value)

    @property
    def cost(self) -> Optional[TokenBalance]:
        """Return the execution cost."""
        return self._cost

    @cost.setter
    def cost(self, value: TokenBalance) -> None:
        self._cost = value

    @property
    def tokens(self) -> set[Token]:
        """Return the pool tokens."""
        return {self.balance1.token, self.balance2.token}

    @property
    def sorted_tokens(self) -> list[Token]:
        """Return a tuple with both tokens sorted."""
        return list(sorted(self.tokens))

    @property
    def balance_update1(self) -> TokenBalance:
        """Return the traded amount of the first token."""
        return self._balance_update1

    @balance_update1.setter
    def balance_update1(self, value: Optional[Amount]) -> None:
        # TODO - this should be update
        self._balance_update1 = TokenBalance.parse_amount(value, self.balance1.token)

    @property
    def balance_update2(self) -> TokenBalance:
        """Return the traded amount of the second token."""
        return self._balance_update2

    @balance_update2.setter
    def balance_update2(self, value: Optional[Amount]) -> None:
        self._balance_update2 = TokenBalance.parse_amount(value, self.balance2.token)

    @property
    def exec_plan_coords(self) -> ExecPlanCoords:
        """Return the coordinates of this uniswap in the execution plan."""
        return self._exec_plan_coords

    def other_token(self, token: Token) -> Token:
        """Returns the "other" token that is not token."""
        assert token in self.tokens
        return (self.tokens - {token}).pop()

    #####################
    #  UTILITY METHODS  #
    #####################

    def balance(self, token: Token) -> TokenBalance:
        """Get the current balance of the given token.
        Args:
            token: A Token.

        Returns: The TokenBalance of the given token.
        """

        if token == self.balance1.token:
            return self.balance1

        if token == self.balance2.token:
            return self.balance2

        raise ValueError(f"Token {token} is not part of AMM {self.pool_id}!")

    def execute(
        self,
        b1_update: NumericType,
        b2_update: NumericType,
        amount_tol: Decimal = Decimal("1e-8"),
    ) -> None:
        """Execute the uniswap at given amounts.

        Args:
            b1_update: Traded amount of token1.
            b2_update: Traded amount of token2.
            amount_tol: Accepted violation of the limit buy/sell amount constraints.
        """
        assert isinstance(b1_update, (int, float, Decimal))
        assert isinstance(b2_update, (int, float, Decimal))

        token1 = self.balance1.token
        token2 = self.balance2.token

        balance_update1 = TokenBalance(b1_update, token1)
        balance_update2 = TokenBalance(b2_update, token2)

        # if any amount is very small, set to zero.
        if any(
            [
                abs(balance_update1) <= TokenBalance(amount_tol, token1),
                abs(balance_update2) <= TokenBalance(amount_tol, token2),
            ]
        ):
            balance_update1 = TokenBalance(0.0, token1)
            balance_update2 = TokenBalance(0.0, token2)

        # Store execution information.
        self.balance_update1 = balance_update1
        self.balance_update2 = balance_update2

    def is_executed(self) -> bool:
        """True if amm is executed otherwise false"""
        return not self.balance_update1.is_zero() and not self.balance_update2.is_zero()

    def get_marginal_xrate(self) -> XRate:
        """Derive the marginal exchange rate from the pool balances."""
        return XRate(self.balance1, self.balance2)

    def is_valid(self) -> bool:
        """Validate the pool."""
        if not isinstance(self.balance1, TokenBalance) or self.balance1 < 0:
            return False
        if not isinstance(self.balance2, TokenBalance) or self.balance2 < 0:
            return False

        return True

    def __str__(self) -> str:
        """Represent as string."""
        return json.dumps(self.as_dict(), indent=2)

    def __repr__(self) -> str:
        """Represent as short string."""
        return f"u{self.pool_id}"

    def __hash__(self) -> int:
        return hash(self.pool_id)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Uniswap):
            return NotImplemented
        if self.pool_id != other.pool_id:
            return False
        return True

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, Uniswap):
            return NotImplemented
        return self.pool_id < other.pool_id


def serialize_amms(amms: list[Uniswap]) -> UniswapsSerializedType:
    """Return AMMs as dict of dicts."""
    return {amm.pool_id: amm.as_dict() for amm in amms if amm.kind != AMMKind.UNISWAP}

"""Representation of Uniswap pool."""
from __future__ import annotations

import json
import logging
from decimal import Decimal
from typing import Optional, Any, Union

from src.models.exchange_rate import ExchangeRate as XRate
from src.models.token import Token, TokenBalance
from src.models.types import NumericType
from src.util.enums import AMMKind
from src.util.exec_plan_coords import ExecPlanCoords
from src.util.numbers import decimal_to_str

FeeType = Union[float, Decimal]
UniswapSerializedType = dict[str, Any]
UniswapsSerializedType = dict[str, UniswapSerializedType]


class Uniswap:
    """Representation of an Automated Market Maker.

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
        weight: float = 1,
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
        # Consistency checks.
        if balance1.token == balance2.token:
            logging.error("Pool tokens cannot be equal!")
            raise ValueError

        if not all(tb.is_positive() for tb in [balance1, balance2]):
            message = f"Uniswap <{pool_id}>: balance1={balance1} balance2={balance2}"
            logging.error(message)
            raise ValueError("Both token balances must be positive!")

        # Store given pool pool_id.
        self.pool_id = pool_id
        self.balance1 = balance1
        self.balance2 = balance2
        self.fee = fee if isinstance(fee, Decimal) else Decimal(fee)
        self.cost = cost
        self.mandatory = mandatory
        self.kind = kind
        self.weight = weight

        self._balance_update1: Optional[TokenBalance] = None
        self._balance_update2: Optional[TokenBalance] = None
        self.exec_plan_coords: Optional[ExecPlanCoords] = None

    @classmethod
    def from_dict(
        cls, amm_id: str, amm_data: UniswapSerializedType
    ) -> Optional[Uniswap]:
        """Construct AMM object from data dict.
        NOTE: Currently, the code only supports Uniswap-style AMMs, i.e.,
        constant-product pools with two tokens and equal weights.
        Args:
            amm_id: AMM pool_id.
            amm_data: Dict of uniswap data.
        Returns:
            A Uniswap object.
        """
        for attr in ["kind", "reserves", "fee"]:
            if attr not in amm_data:
                raise ValueError(f"Missing field '{attr}' in amm <{amm_id}>!")

        kind = AMMKind(amm_data["kind"])
        reserves = amm_data.get("reserves")
        weight = 0.5

        if kind == AMMKind.CONSTANT_PRODUCT:
            # Parse UniswapV2/Sushiswap pools.
            if not isinstance(reserves, dict):
                raise ValueError(
                    f"AMM <{amm_id}>: 'reserves' must be a dict of Token -> amount!"
                )
            if len(reserves) != 2:
                message = (
                    f"AMM <{amm_id}>: "
                    f"ConstantProduct AMMs are only supported with 2 tokens!"
                )
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

            weight = list(reserves.values())[0]["weight"]
            balance1, balance2 = [
                TokenBalance(Decimal(b["balance"]), Token(t))
                for t, b in reserves.items()
            ]

        else:
            logging.warning(
                f"AMM <{amm_id}>: type <{kind}> is currently not supported!"
            )
            return None

        return Uniswap(
            pool_id=amm_id,
            balance1=balance1,
            balance2=balance2,
            fee=Decimal(amm_data["fee"]),
            cost=TokenBalance.parse(amm_data.get("cost"), allow_none=True),
            kind=kind,
            weight=weight,
        )

    def as_dict(self) -> dict:
        """Return AMM object as dictionary.

        NOTE: Currently, the code only supports Uniswap-style AMMs, i.e.,
        constant-product pools with two tokens and equal weights.

        """
        token1 = str(self.token1)
        token2 = str(self.token2)
        balance1 = decimal_to_str(self.balance1.as_decimal())
        balance2 = decimal_to_str(self.balance2.as_decimal())

        reserves: Union[str, dict]
        if self.kind == AMMKind.WEIGHTED_PRODUCT:
            reserves = {
                token1: {"balance": balance1, "weight": self.weight},
                token2: {"balance": balance2, "weight": self.weight},
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
            assert self.balance_update1 is not None and self.balance_update2 is not None
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

            if self.exec_plan_coords is None:
                logging.warning(
                    f"AMM <{self.pool_id}>: "
                    f"has balance updates with invalid execution plan"
                )
                exec_plan = None
            else:
                exec_plan = self.exec_plan_coords.as_dict()

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
    def token1(self) -> Token:
        """Returns token1"""
        return self.balance1.token

    @property
    def token2(self) -> Token:
        """Returns token2"""
        return self.balance2.token

    @property
    def tokens(self) -> set[Token]:
        """Return the pool tokens."""
        return {self.balance1.token, self.balance2.token}

    @property
    def balance_update1(self) -> Optional[TokenBalance]:
        """Return the traded amount of the first token."""
        return self._balance_update1

    @property
    def balance_update2(self) -> Optional[TokenBalance]:
        """Return the traded amount of the second token."""
        return self._balance_update2

    def other_token(self, token: Token) -> Token:
        """Returns the "other" token that is not token."""
        assert token in self.tokens
        return (self.tokens - {token}).pop()

    #####################
    #  UTILITY METHODS  #
    #####################

    def execute(
        self,
        b1_update: NumericType,
        b2_update: NumericType,
    ) -> None:
        """Execute the uniswap at given amounts.

        Args:
            b1_update: Traded amount of token1.
            b2_update: Traded amount of token2.
        """
        assert isinstance(b1_update, (int, float, Decimal))
        assert isinstance(b2_update, (int, float, Decimal))

        # Store execution information.
        self._balance_update1 = TokenBalance(b1_update, self.token1)
        self._balance_update2 = TokenBalance(b2_update, self.token2)

    def is_executed(self) -> bool:
        """True if amm is executed otherwise false"""
        return self.balance_update1 is not None and self.balance_update2 is not None

    def get_marginal_xrate(self) -> XRate:
        """Derive the marginal exchange rate from the pool balances."""
        return XRate(self.balance1, self.balance2)

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

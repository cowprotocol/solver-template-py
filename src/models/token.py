"""A class for extendable token enum's."""
from __future__ import annotations

import re
from decimal import Decimal, getcontext
from typing import Optional, Union

from src.models.types import NumericType
from src.util.constants import Constants


class Token:
    """Enumeration over available tokens."""

    def __init__(self, value: str):
        if Token._is_valid(value):
            self.value = value
        else:
            raise ValueError(f"Invalid Ethereum Address {value}")

    @staticmethod
    def _is_valid(address: str) -> bool:
        match_result = re.match(
            pattern=r"^(0x)?[0-9a-f]{40}$", string=address, flags=re.IGNORECASE
        )
        return match_result is not None

    def __str__(self) -> str:
        """Convert to string."""
        return self.value

    def __repr__(self) -> str:
        """Convert to string."""
        return self.__str__()

    def __hash__(self) -> int:
        """Hash of token."""
        return hash(self.value)

    def __eq__(self, other: object) -> bool:
        """Equality operator."""
        if isinstance(other, Token):
            return self.value == other.value
        return False

    def __lt__(self, other: object) -> bool:
        """Less-than operator."""
        if isinstance(other, Token):
            return self.value < other.value
        return NotImplemented


class TokenInfo:
    """Class for storing token information."""

    def __init__(
        self,
        token: Token,
        decimals: int,
        alias: Optional[str] = None,
        external_price: Optional[Decimal] = None,
        estimated_price: Optional[Decimal] = None,
        internal_buffer: Optional[Decimal] = None,
        normalize_priority: Optional[int] = 0,
    ):
        """Constructor."""
        self.token = token
        self.alias = alias
        self.decimals = decimals
        self.external_price = external_price
        self.estimated_price = estimated_price
        self.internal_buffer = internal_buffer
        self._normalize_priority = normalize_priority or 0

    @property
    def normalize_priority(self) -> int:
        """
        Return the token priority for normalization purposes.

        Higher value means higher priority.
        """
        return self._normalize_priority

    def as_dict(self) -> dict:
        """Convert to dict."""
        attr = [
            a
            for a in dir(self)
            if not callable(getattr(self, a)) and not a.startswith("_") and a != "token"
        ]

        return {a: getattr(self, a) for a in sorted(attr)}

    def __str__(self) -> str:
        """Convert to string."""
        token_info_dict = self.as_dict()

        _str = f"Token [{self.token}]:"
        for attr, value in token_info_dict.items():
            if isinstance(value, Decimal) and attr not in [
                "external_price",
                "internal_buffer",
            ]:
                value = value.quantize(Constants.DECIMAL_STR_PREC)
            _str += f"\n-- {attr} : {value}"

        return _str


def select_token_with_highest_normalize_priority(
    tokens: dict[Token, TokenInfo]
) -> Token:
    """
    Select token with highest normalize priority from the list of tokens.

    If the highest normalize_priority is shared by multiple tokens, the
    ref_token is the first lexicographically.
    """
    max_priority = max(t.normalize_priority for t in tokens.values())
    highest_priority_tokens = [
        t for t, info in tokens.items() if info.normalize_priority == max_priority
    ]
    return highest_priority_tokens[0]


TokenDict = dict[Token, TokenInfo]
TokenSerializedType = str
TokenAmountSerializedType = tuple[Union[str, NumericType], TokenSerializedType]


class TokenBalance:
    """Class to represent an amount of some token."""

    def __init__(self, balance: NumericType, token: Token):
        """Initialize.

        Args:
            balance: Amount of tokens.
            token: Token.
        """

        self._balance = Decimal(balance)
        self.balance = balance
        self.token = token

        if not self._balance.is_finite():
            raise ValueError(f"Token balance must be finite, not {self._balance}!")

    @classmethod
    def parse(
        cls,
        token_amount_serialized: Optional[TokenAmountSerializedType],
        allow_negative: bool = False,
        allow_none: bool = False,
    ) -> Optional[TokenBalance]:
        """
        Method to parse a token amount given as (amount, token) into a TokenBalance.
        """
        if token_amount_serialized is None:
            if not allow_none:
                raise ValueError("Token amount must not be None!")
            token_amount = None
        else:
            if not isinstance(token_amount_serialized, dict) or set(
                token_amount_serialized.keys()
            ) != {"amount", "token"}:
                raise ValueError(
                    "token amount must be given as dict of {'amount': .., 'token': ..},"
                    f" not <{token_amount_serialized}>!"
                )
            token_amount = cls(
                Decimal(token_amount_serialized["amount"]),
                Token(token_amount_serialized["token"]),
            )
            if not allow_negative and token_amount.is_negative():
                raise ValueError(f"Token amount must be non-negative ({token_amount})!")
        return token_amount

    @classmethod
    def parse_amount(
        cls, amt_type: Optional[Amount], token: Token
    ) -> Optional[TokenBalance]:
        """Auxiliary method to parse a numerical value into a TokenBalance.

        Args:
            amt_type: Amount to be set, or None.
            token: Token belonging to amount.

        Returns:
            A TokenBalance, or None.

        """

        if isinstance(amt_type, (int, float, Decimal)):
            return cls(amt_type, token)

        if isinstance(amt_type, TokenBalance):
            if amt_type.token != token:
                raise ValueError(
                    f"Tokens do not match: <{amt_type.token}> vs. <{token}>!"
                )
            return amt_type

        return None

    def as_decimal(self) -> Decimal:
        """Returns balance attribute as Decimal type"""
        return self._balance

    @staticmethod
    def precision() -> int:
        """Return precision currently associated with TokenBalance."""
        return getcontext().prec

    def is_positive(self) -> bool:
        """Determine if a TokenBalance is positive."""
        return self._balance > 0

    def is_negative(self) -> bool:
        """Determine if a TokenBalance is negative."""
        return self._balance < 0

    def is_zero(self) -> bool:
        """Determine if a TokenBalance is zero."""
        return self._balance == 0

    def __eq__(self, other: object) -> bool:
        """Equality operator.

        Args:
            other: Another TokenBalance, or zero.
        """
        if other == 0:
            return self.is_zero()

        if isinstance(other, TokenBalance):

            if self.token != other.token:
                raise ValueError(
                    f"Cannot compare different tokens <{self.token}> / <{other.token}>!"
                )
            return self._balance == other._balance

        raise ValueError(f"Cannot compare TokenBalance and type <{type(other)}>!")

    def __ne__(self, other: object) -> bool:
        """Non-equality operator"""
        return not self == other

    def __lt__(self, other: Union[TokenBalance, NumericType]) -> bool:
        """Less-than operator.

        Args:
            other: Another TokenBalance, or zero.
        """
        if isinstance(other, TokenBalance):

            if self.token != other.token:
                raise ValueError(
                    f"Cannot compare different tokens <{self.token}> / <{other.token}>!"
                )
            return self._balance < other._balance

        if other == 0:
            return self._balance < 0

        raise ValueError(f"Cannot compare TokenBalance and type <{type(other)}>")

    def __le__(self, other: Union[TokenBalance, NumericType]) -> bool:
        """Less-than-or-equal operator.

        Args:
            other: Another TokenBalance, or zero.
        """
        return self < other or self == other

    def __gt__(self, other: Union[TokenBalance, NumericType]) -> bool:
        """Greater-than operator.

        Args:
            other: Another TokenBalance, or zero.
        """
        return not self <= other

    def __ge__(self, other: Union[TokenBalance, NumericType]) -> bool:
        """Greater-than-or-equal operator.

        Args:
            other: Another TokenBalance, or zero.
        """
        return self > other or self == other

    def __neg__(self) -> TokenBalance:
        """Negation operator."""
        return TokenBalance(-self._balance, self.token)

    def __abs__(self) -> TokenBalance:
        """Absolute value operator."""
        return TokenBalance(abs(self._balance), self.token)

    def __add__(self, other: Union[TokenBalance, NumericType]) -> TokenBalance:
        """Addition operator.

        Args:
            other: Another TokenBalance, or zero.
        """
        if isinstance(other, TokenBalance):

            if self.token == other.token:
                return TokenBalance(self._balance + other._balance, self.token)

            raise ValueError(f"Cannot add <{other.token}> and <{self.token}>!")

        if other == 0:
            # This is required to enable the use of the sum() function:
            # sum() by design starts with a value of '0' and then iteratively
            # adds the items in the list that is passed as argument. See:
            # https://stackoverflow.com/questions/1218710/pythons-sum-and-non-integer-values
            return self

        raise ValueError(f"Cannot add <{type(other)}> and TokenBalance!")

    def __radd__(self, other: Union[TokenBalance, NumericType]) -> TokenBalance:
        """Addition-from-right operator.

        Args:
            other: Another TokenBalance, or zero.
        """
        return self + other

    def __sub__(self, other: Union[TokenBalance, NumericType]) -> TokenBalance:
        """Subtraction operator.

        Args:
            other: Another TokenBalance, or zero.
        """
        return self + (-other)

    def __rsub__(self, other: Union[TokenBalance, NumericType]) -> TokenBalance:
        """Subtraction operator.

        Args:
            other: Another TokenBalance, or zero.
        """
        return other + (-self)

    def __mul__(self, other: NumericType) -> TokenBalance:
        """Multiplication operator.

        Args:
            other: A {float|int|Decimal}.
        """
        if isinstance(other, (int, float, Decimal)):
            return TokenBalance(Decimal(other) * self._balance, self.token)

        raise ValueError(f"Cannot multiply TokenBalance by <{type(other)}>!")

    def __rmul__(self, other: NumericType) -> TokenBalance:
        """Multiplication-from-right operator.

        Args:
            other: A {float|int|Decimal}.
        """
        return self * other

    def __truediv__(self, other: Union[TokenBalance, NumericType]) -> TokenBalance:
        """Division operator.

        Args:
            other: A {TokenBalance|float|int|Decimal}.
        """
        if isinstance(other, (int, float, Decimal)):
            if other == 0:
                raise ZeroDivisionError
            return TokenBalance(self._balance / Decimal(other), self.token)

        if isinstance(other, TokenBalance):
            if self.token == other.token:
                return TokenBalance(self._balance / other._balance, self.token)
            raise ValueError(
                f"Can't divide TokenBalances with different "
                f"tokens <{self.token}> and <{other.token}>!"
            )

        raise ValueError(f"Cannot divide TokenBalance by <{type(other)}>!")

    def __rtruediv__(self, other: object) -> None:
        """Division-from-right operator.

        Args:
            other: Something.
        """
        raise ValueError(f"<{type(other)}> cannot be divided by TokenBalance!")

    def __str__(self) -> str:
        """Represent as string (rounded to 5 decimals)."""
        return f"{self.token}: {self.balance}"

    def __repr__(self) -> str:
        """Represent as string."""
        return str(self)


Amount = Union[TokenBalance, NumericType]

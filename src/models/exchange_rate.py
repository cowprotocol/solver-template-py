"""Representation of an exchange rate between two tokens."""
from __future__ import annotations

from src.models.token import Token, TokenBalance
from src.models.types import NumericType


class ExchangeRate:
    """Class representing the exchange rate between two tokens."""

    def __init__(self, tb1: TokenBalance, tb2: TokenBalance):
        """
        An ExchangeRate is represented as equivalence of two TokenBalances,
        e.g., 2 [ETH] == 800 [EUR] --> 400 [EUR]/[ETH] or 0.0025 [ETH]/[EUR].
        Args:
            tb1: First TokenBalance.
            tb2: Second TokenBalance.
        """
        assert isinstance(tb1, TokenBalance)
        assert isinstance(tb2, TokenBalance)

        if tb1.token == tb2.token:
            raise ValueError("Both given tokens are identical!")

        if not (tb1.is_positive() and tb2.is_positive()):
            raise ValueError(f"Both token balances must be positive! {tb1} & {tb2}")

        # Store attributes.
        self.tb1 = tb1
        self.tb2 = tb2

    @classmethod
    def from_prices(
        cls,
        token1_price: tuple[Token, NumericType],
        token2_price: tuple[Token, NumericType],
    ) -> ExchangeRate:
        """Alternative constructor: Build ExchangeRate from (absolute) token prices.

        Args:
            token1_price: Tuple of (Token, price).
            token2_price: Tuple of (Token, price).
        Returns:
            The ExchangeRate between the input tokens at the given prices.
        """
        # Validate Input
        balances = []
        for token_price in [token1_price, token2_price]:
            assert isinstance(token_price, tuple) and len(token_price) == 2
            token, price = token_price
            assert isinstance(token, Token)
            assert price > 0

            balances.append(TokenBalance(price, token))

        return cls(balances[0], balances[1])

    def token_balance(self, token: Token) -> TokenBalance:
        """Get token balance for given token."""
        if token == self.tb1.token:
            return self.tb1
        if token == self.tb2.token:
            return self.tb2

        raise ValueError(f"Exchange rate does not involve {token}")

    @property
    def tokens(self) -> set[Token]:
        """Returns a set containing the two tokens."""
        return {self.tb1.token, self.tb2.token}

    def convert(self, token_balance: TokenBalance) -> TokenBalance:
        """Convert a TokenBalance of one token into a TokenBalance of the other.
        Args:
            token_balance: TokenBalance to be converted.
        Returns: Converted TokenBalance.
        """
        assert isinstance(token_balance, TokenBalance)

        if token_balance.token == self.tb1.token:
            # Division of two token balances with same token yields scalar
            # which can then be multiplied with another token balance.
            return (token_balance / self.tb1).as_decimal() * self.tb2

        if token_balance.token == self.tb2.token:
            return (token_balance / self.tb2).as_decimal() * self.tb1

        raise ValueError(
            f"Token balance <{token_balance}> can not be "
            f"converted using ExchangeRate <{self.tb1.token}/{self.tb2.token}>!"
        )

    def convert_unit(self, token: Token) -> TokenBalance:
        """Convert one unit of one token into a TokenBalance of the other.
        Args:
            token: Token to be converted.
        Returns:
            Converted TokenBalance.
        """
        assert isinstance(token, Token)
        return self.convert(TokenBalance(1, token))

    def __eq__(self, other: object) -> bool:
        """Equality operator"""
        if not isinstance(other, ExchangeRate):
            raise ValueError(f"Cannot compare ExchangeRate and type <{type(other)}>!")

        # The ratio of the TokenBalances must be equal.
        return self.convert(other.tb1) == other.tb2

    def __ne__(self, other: object) -> bool:
        """Non-equality operator"""
        return not self == other

    def __str__(self) -> str:
        """Represent as string."""
        tb1 = self.convert(TokenBalance(1, self.tb1.token))
        tb2 = self.convert(TokenBalance(1, self.tb2.token))
        return f"{tb1}/[{self.tb1.token}]  <=>  {tb2}/[{self.tb2.token}]"

    def __repr__(self) -> str:
        """Represent as string."""
        return str(self)

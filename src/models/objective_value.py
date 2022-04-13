"""Data structure for evaluating objective"""
from __future__ import annotations
from dataclasses import dataclass
from decimal import Decimal

from src.models.token import Token


@dataclass
class ObjectiveValue:
    """Simplifies return type of Order.evaluate_objective"""

    ref_token: Token
    ref_token_price: Decimal
    volume: Decimal
    surplus: Decimal
    fees: Decimal
    cost: Decimal

    @classmethod
    def zero(cls, ref_token: Token, price: Decimal) -> ObjectiveValue:
        """Default Objective value of zero in all fields for given ref_token"""
        return cls(
            ref_token=ref_token,
            ref_token_price=price,
            volume=Decimal(0),
            surplus=Decimal(0),
            fees=Decimal(0),
            cost=Decimal(0),
        )

    def surplus_plus_fees_minus_cost(self) -> Decimal:
        """Returns the objective Value as surplus + fee - cost"""
        return self.surplus + self.fees - self.cost

    def __add__(self, other: object) -> ObjectiveValue:
        if isinstance(other, ObjectiveValue):
            if self.ref_token != other.ref_token:
                raise ValueError("Cant add with different reference tokens!")
            return ObjectiveValue(
                ref_token=self.ref_token,
                ref_token_price=self.ref_token_price,
                volume=self.volume + other.volume,
                surplus=self.surplus + other.surplus,
                fees=self.fees + other.fees,
                cost=self.cost + other.cost,
            )
        raise ValueError(f"Cant add object of type {type(other)}")

    def ref_token_volume(self, amount: Decimal, price: Decimal) -> Decimal:
        """Computes volume of a value relative in ref token"""
        value = amount * price
        return value / self.ref_token_price

"""
Liquidity domain model for CoW Protocol.

This module contains liquidity models for different AMM types.
"""

from typing import Dict, Optional
from pydantic import BaseModel, Field, field_validator


class TokenPair(BaseModel):
    """Token pair for liquidity."""

    token_a: str = Field(..., description="First token address")
    token_b: str = Field(..., description="Second token address")


class ScalingFactor(BaseModel):
    """Scaling factor for liquidity calculations."""

    factor: int = Field(..., description="Scaling factor")


class LiquidityState(BaseModel):
    """Liquidity state information."""

    id: str = Field(..., description="Liquidity source identifier")
    state: Dict[str, int] = Field(..., description="State variables")
    token_pair: TokenPair = Field(..., description="Token pair")
    scaling_factor: Optional[ScalingFactor] = Field(None, description="Scaling factor")


class ConstantProductLiquidity(BaseModel):
    """Constant product liquidity (Uniswap V2 style)."""

    id: str = Field(..., description="Liquidity source identifier")
    tokens: Dict[str, int] = Field(..., description="Token balances")
    fee: int = Field(..., description="Trading fee in basis points")


class WeightedProductLiquidity(BaseModel):
    """Weighted product liquidity (Balancer style)."""

    id: str = Field(..., description="Liquidity source identifier")
    tokens: Dict[str, int] = Field(..., description="Token balances")
    weights: Dict[str, int] = Field(..., description="Token weights")
    fee: int = Field(..., description="Trading fee in basis points")


class StableLiquidity(BaseModel):
    """Stable liquidity (Curve style)."""

    id: str = Field(..., description="Liquidity source identifier")
    tokens: Dict[str, int] = Field(..., description="Token balances")
    amplification: int = Field(..., description="Amplification parameter")
    fee: int = Field(..., description="Trading fee in basis points")


class ConcentratedLiquidity(BaseModel):
    """Concentrated liquidity (Uniswap V3 style)."""

    id: str = Field(..., description="Liquidity source identifier")
    token_a: str = Field(..., description="First token address")
    token_b: str = Field(..., description="Second token address")
    liquidity: int = Field(..., description="Liquidity amount")
    tick_lower: int = Field(..., description="Lower tick")
    tick_upper: int = Field(..., description="Upper tick")
    fee: int = Field(..., description="Trading fee in basis points")


class LimitOrderLiquidity(BaseModel):
    """Limit order liquidity."""

    id: str = Field(..., description="Liquidity source identifier")
    order_uid: str = Field(..., description="Order identifier")
    sell_token: str = Field(..., description="Token to sell")
    buy_token: str = Field(..., description="Token to buy")
    sell_amount: int = Field(..., description="Amount to sell in wei")
    buy_amount: int = Field(..., description="Amount to buy in wei")
    fee_amount: int = Field(..., description="Fee amount in wei")


class Liquidity(BaseModel):
    """Generic liquidity source."""

    id: str = Field(..., description="Liquidity source identifier")
    kind: str = Field(..., description="Liquidity type")
    tokens: Dict[str, int] = Field(..., description="Token balances")

    # Optional fields for specific liquidity types
    constant_product: Optional[ConstantProductLiquidity] = Field(
        None, description="Constant product liquidity"
    )
    weighted_product: Optional[WeightedProductLiquidity] = Field(
        None, description="Weighted product liquidity"
    )
    stable: Optional[StableLiquidity] = Field(None, description="Stable liquidity")
    concentrated_liquidity: Optional[ConcentratedLiquidity] = Field(
        None, description="Concentrated liquidity"
    )
    limit_order: Optional[LimitOrderLiquidity] = Field(
        None, description="Limit order liquidity"
    )

    @field_validator("kind")
    @classmethod
    def validate_kind(cls, v):
        valid_kinds = [
            "constant_product",
            "weighted_product",
            "stable",
            "concentrated_liquidity",
            "limit_order",
        ]
        if v not in valid_kinds:
            raise ValueError(f"kind must be one of {valid_kinds}")
        return v

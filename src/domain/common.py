"""
Common types and utilities for CoW Protocol models.
"""

from typing import Union
from decimal import Decimal
from pydantic import BaseModel, Field, validator


class Wei(BaseModel):
    """Wei amount (smallest unit of ETH)."""

    value: int = Field(..., description="Wei amount")

    @validator("value")
    def validate_positive(cls, v):
        if v < 0:
            raise ValueError("Wei amount must be non-negative")
        return v


class GasPrice(BaseModel):
    """Gas price in wei per gas."""

    value: int = Field(..., description="Gas price in wei")

    @validator("value")
    def validate_positive(cls, v):
        if v < 0:
            raise ValueError("Gas price must be non-negative")
        return v


class Price(BaseModel):
    """Token price."""

    value: int = Field(..., description="Price in wei")
    denominator: int = Field(..., description="Price denominator")

    @validator("value")
    def validate_positive(cls, v):
        if v < 0:
            raise ValueError("Price must be non-negative")
        return v

    @validator("denominator")
    def validate_denominator_positive(cls, v):
        if v <= 0:
            raise ValueError("Price denominator must be positive")
        return v


class Id(BaseModel):
    """Identifier with validation."""

    value: str = Field(..., description="Identifier value")

    @validator("value")
    def validate_format(cls, v):
        if not v.startswith("0x"):
            raise ValueError("ID must start with 0x")
        if len(v) != 66:  # 0x + 64 hex chars
            raise ValueError("ID must be 66 characters long")
        return v


class Deadline(BaseModel):
    """Auction deadline."""

    value: str = Field(..., description="Deadline in ISO 8601 format")

    @validator("value")
    def validate_iso_format(cls, v):
        # Basic ISO 8601 validation
        if "T" not in v:
            raise ValueError("Deadline must be in ISO 8601 format")
        return v


class TokenId(BaseModel):
    """Token identifier."""

    address: str = Field(..., description="Token contract address")


class OrderUid(BaseModel):
    """Order unique identifier."""

    value: str = Field(..., description="Order UID")

    @validator("value")
    def validate_format(cls, v):
        if not v.startswith("0x"):
            raise ValueError("Order UID must start with 0x")
        if len(v) != 66:  # 0x + 64 hex chars
            raise ValueError("Order UID must be 66 characters long")
        return v


class LiquidityId(BaseModel):
    """Liquidity source identifier."""

    value: str = Field(..., description="Liquidity ID")

    @validator("value")
    def validate_not_empty(cls, v):
        if not v:
            raise ValueError("Liquidity ID cannot be empty")
        return v


class ScalingFactor(BaseModel):
    """Scaling factor for calculations."""

    value: int = Field(..., description="Scaling factor")

    @validator("value")
    def validate_positive(cls, v):
        if v <= 0:
            raise ValueError("Scaling factor must be positive")
        return v

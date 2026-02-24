"""
Order domain model for CoW Protocol.

This module contains the Order model and related types.
"""

from typing import Optional, Literal
from pydantic import BaseModel, Field, validator


class OrderSide:
    """Order side enumeration."""

    SELL = "sell"
    BUY = "buy"


class OrderClass:
    """Order class enumeration."""

    MARKET = "market"
    LIMIT = "limit"
    LIQUIDITY = "liquidity"


class EcdsaSignature(BaseModel):
    """ECDSA signature."""

    v: int = Field(..., ge=0, le=255, description="Recovery ID")
    r: int = Field(..., description="R component")
    s: int = Field(..., description="S component")


class AppData(BaseModel):
    """Application data hash."""

    hash: str = Field(..., description="App data hash")


class FlashloanHint(BaseModel):
    """Flashloan hint for order execution."""

    tokens: list[str] = Field(..., description="Tokens for flashloan")


class Order(BaseModel):
    """CoW Protocol order."""

    uid: str = Field(..., description="Unique order identifier")
    sell_token: str = Field(..., description="Token to sell")
    buy_token: str = Field(..., description="Token to buy")
    sell_amount: int = Field(..., description="Amount to sell in wei")
    buy_amount: int = Field(..., description="Amount to buy in wei")
    fee_amount: int = Field(..., description="Fee amount in wei")
    kind: Literal["sell", "buy"] = Field(..., description="Order kind")
    partially_fillable: bool = Field(
        False, description="Whether order can be partially filled"
    )
    class_: Literal["market", "limit", "liquidity"] = Field(
        ..., alias="class", description="Order class"
    )
    signature: str = Field(..., description="Order signature")
    app_data: str = Field(..., description="Application data hash")
    flashloan_hint: Optional[FlashloanHint] = Field(None, description="Flashloan hint")

    @validator("uid")
    def validate_uid_format(cls, v):
        if not v.startswith("0x"):
            raise ValueError("uid must start with 0x")
        if len(v) != 66:  # 0x + 64 hex chars
            raise ValueError("uid must be 66 characters long")
        return v

    @validator("signature")
    def validate_signature_format(cls, v):
        if not v.startswith("0x"):
            raise ValueError("signature must start with 0x")
        return v

    @validator("app_data")
    def validate_app_data_format(cls, v):
        if not v.startswith("0x"):
            raise ValueError("app_data must start with 0x")
        return v


class JitOrder(BaseModel):
    """Just-in-time order."""

    order: Order = Field(..., description="The JIT order")
    signature: str = Field(..., description="JIT order signature")

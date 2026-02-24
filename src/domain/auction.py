"""
Auction domain model for CoW Protocol.

This module contains the Auction model and related types.
"""

from typing import Dict, List, Optional, Union
from decimal import Decimal
from pydantic import BaseModel, Field, field_validator


class Token(BaseModel):
    """Token information."""

    decimals: Optional[int] = Field(None, description="Token decimals")
    symbol: Optional[str] = Field(None, description="Token symbol")
    reference_price: Optional[str] = Field(None, description="Reference price in wei")
    available_balance: Optional[str] = Field(None, description="Available balance")
    trusted: Optional[bool] = Field(False, description="Trusted token")

    model_config = {"extra": "ignore"}


class Order(BaseModel):
    """Order in an auction."""

    uid: str = Field(..., description="Unique order identifier")
    sell_token: str = Field(..., alias="sellToken", description="Token to sell")
    buy_token: str = Field(..., alias="buyToken", description="Token to buy")
    sell_amount: int = Field(
        ..., alias="sellAmount", description="Amount to sell in wei"
    )
    full_sell_amount: int = Field(
        ..., alias="fullSellAmount", description="Full sell amount in wei"
    )
    buy_amount: int = Field(..., alias="buyAmount", description="Amount to buy in wei")
    full_buy_amount: int = Field(
        ..., alias="fullBuyAmount", description="Full buy amount in wei"
    )
    valid_to: int = Field(..., alias="validTo", description="Valid until timestamp")
    kind: str = Field(..., description="Order kind")
    receiver: str = Field(..., description="Order receiver")
    owner: str = Field(..., description="Order owner")
    partially_fillable: bool = Field(
        False,
        alias="partiallyFillable",
        description="Whether order can be partially filled",
    )
    pre_interactions: List[dict] = Field(
        default_factory=list, alias="preInteractions", description="Pre-interactions"
    )
    post_interactions: List[dict] = Field(
        default_factory=list, alias="postInteractions", description="Post-interactions"
    )
    sell_token_source: str = Field(
        "erc20", alias="sellTokenSource", description="Sell token source"
    )
    buy_token_destination: str = Field(
        "erc20", alias="buyTokenDestination", description="Buy token destination"
    )
    order_class: str = Field("limit", alias="class", description="Order class")
    app_data: str = Field("", alias="appData", description="Application data hash")
    signing_scheme: str = Field(
        "eip712", alias="signingScheme", description="Signing scheme"
    )
    signature: str = Field(..., description="Order signature")
    fee_amount: int = Field(
        0, alias="feeAmount", description="Fee amount in wei (deprecated, defaults to 0)"
    )

    model_config = {"extra": "ignore", "populate_by_name": True}


class Liquidity(BaseModel):
    """Liquidity source in an auction."""

    kind: str = Field(..., description="Liquidity type")
    id: str = Field(..., description="Liquidity source identifier")
    address: str = Field(..., description="Liquidity source address")
    router: str = Field(..., description="Router address")
    gas_estimate: int = Field(..., alias="gasEstimate", description="Gas estimate")
    tokens: Dict[str, Dict[str, str]] = Field(
        default_factory=dict, description="Token balances"
    )
    fee: int = Field(..., description="Trading fee")

    @field_validator("fee", mode="before")
    @classmethod
    def validate_fee(cls, v):
        if isinstance(v, str):
            return int(float(v) * 10000)  # Simple: 0.003 -> 30
        return v

    @field_validator("gas_estimate", mode="before")
    @classmethod
    def validate_gas_estimate(cls, v):
        if isinstance(v, str):
            return int(v)
        return v

    model_config = {"extra": "ignore", "populate_by_name": True}


class Auction(BaseModel):
    """CoW Protocol auction."""

    id: str = Field(..., description="Auction identifier")
    tokens: Dict[str, Token] = Field(
        default_factory=dict, description="Token information"
    )
    orders: List[Order] = Field(
        default_factory=list, description="Orders in the auction"
    )
    liquidity: List[Liquidity] = Field(
        default_factory=list, description="Liquidity sources"
    )
    effective_gas_price: int = Field(
        ..., alias="effectiveGasPrice", description="Effective gas price in wei"
    )
    deadline: str = Field(..., description="Auction deadline (ISO 8601)")
    surplus_capturing_jit_order_owners: List[str] = Field(
        default_factory=list,
        alias="surplusCapturingJitOrderOwners",
        description="JIT order owners",
    )

    @field_validator("effective_gas_price", mode="before")
    @classmethod
    def validate_effective_gas_price(cls, v):
        if isinstance(v, str):
            return int(v)
        return v

    model_config = {
        "extra": "ignore",  # Ignore unknown fields
        "populate_by_name": True,
    }

"""
Solution domain model for CoW Protocol.

This module contains the Solution model and related types.
"""

from typing import Dict, List, Optional
from pydantic import BaseModel, Field, field_validator, field_serializer
from hexbytes import HexBytes


class Trade(BaseModel):
    """Trade in a solution (fulfillment format)."""

    kind: str = Field(..., description="Trade kind: 'fulfillment' or 'jit'")
    order: str = Field(..., description="Order UID")
    executed_amount: str = Field(..., alias="executedAmount", description="Executed amount in wei")

    model_config = {"populate_by_name": True}


class Fulfillment(BaseModel):
    """Order fulfillment."""

    order_uid: str = Field(..., alias="orderUid", description="Order identifier")
    executed_amount: str = Field(..., alias="executedAmount", description="Executed amount in wei")

    model_config = {"populate_by_name": True}


class Fee(BaseModel):
    """Fee information."""

    token: str = Field(..., description="Fee token")
    amount: str = Field(..., description="Fee amount in wei")


class JitTrade(BaseModel):
    """Just-in-time trade."""

    order_uid: str = Field(..., alias="orderUid", description="JIT order identifier")
    sell_token: str = Field(..., alias="sellToken", description="Token to sell")
    buy_token: str = Field(..., alias="buyToken", description="Token to buy")
    sell_amount: str = Field(..., alias="sellAmount", description="Amount to sell in wei")
    buy_amount: str = Field(..., alias="buyAmount", description="Amount to buy in wei")
    fee_amount: str = Field(..., alias="feeAmount", description="Fee amount in wei")

    model_config = {"populate_by_name": True}


class Asset(BaseModel):
    """Asset with token and amount."""

    token: str = Field(..., description="Token address")
    amount: str = Field(..., description="Amount in wei")


class InteractionAllowance(BaseModel):
    """Allowance for an interaction."""

    token: str = Field(..., description="Token address")
    spender: str = Field(..., description="Spender address")
    amount: str = Field(..., description="Allowance amount in wei")


class Interaction(BaseModel):
    """
    Smart contract interaction (Custom type).

    Matches the Rust CustomInteraction struct with serde tag "kind": "custom".
    """

    kind: str = Field(default="custom", description="Interaction kind: 'custom' or 'liquidity'")
    internalize: bool = Field(default=False, description="Whether to internalize this interaction")
    target: str = Field(..., description="Target contract address")
    value: str = Field(..., description="ETH value to send")
    call_data: HexBytes = Field(..., alias="callData", description="Call data")
    allowances: List[InteractionAllowance] = Field(default_factory=list, description="Token allowances")
    inputs: List[Asset] = Field(default_factory=list, description="Input assets")
    outputs: List[Asset] = Field(default_factory=list, description="Output assets")

    model_config = {"arbitrary_types_allowed": True, "populate_by_name": True}

    @field_validator("call_data", mode="before")
    @classmethod
    def validate_call_data(cls, v):
        if isinstance(v, str):
            return HexBytes(v)
        return v

    @field_serializer("call_data")
    @classmethod
    def serialize_call_data(cls, v: HexBytes) -> str:
        """Serialize HexBytes to hex string for JSON output."""
        if isinstance(v, HexBytes):
            return v.hex()
        return str(v)


class LiquidityInteraction(BaseModel):
    """Liquidity source interaction."""

    liquidity_id: str = Field(..., alias="liquidityId", description="Liquidity source identifier")
    interaction: Interaction = Field(..., description="Interaction details")

    model_config = {"populate_by_name": True}


class CustomInteraction(BaseModel):
    """Custom interaction."""

    interaction: Interaction = Field(..., description="Interaction details")


class Allowance(BaseModel):
    """Token allowance."""

    token: str = Field(..., description="Token address")
    spender: str = Field(..., description="Spender address")
    amount: str = Field(..., description="Allowance amount in wei")


class ClearingPrices(BaseModel):
    """Clearing prices for tokens."""

    prices: Dict[str, int] = Field(..., description="Token prices in wei")


class Single(BaseModel):
    """Single solution."""

    id: int = Field(..., description="Solution identifier")
    trades: List[Trade] = Field(..., description="Trades in the solution")
    prices: Dict[str, str] = Field(..., description="Clearing prices")
    interactions: List[Interaction] = Field(
        ..., description="Smart contract interactions"
    )
    fulfillments: List[Fulfillment] = Field(
        default_factory=list, description="Order fulfillments"
    )
    fees: List[Fee] = Field(default_factory=list, description="Fees")
    jit_trades: List[JitTrade] = Field(default_factory=list, description="JIT trades")
    liquidity_interactions: List[LiquidityInteraction] = Field(
        default_factory=list, description="Liquidity interactions"
    )
    custom_interactions: List[CustomInteraction] = Field(
        default_factory=list, description="Custom interactions"
    )
    allowances: List[Allowance] = Field(
        default_factory=list, description="Token allowances"
    )


class Solution(BaseModel):
    """CoW Protocol solution."""

    id: int = Field(..., description="Solution identifier")
    trades: List[Trade] = Field(..., description="Trades in the solution")
    prices: Dict[str, str] = Field(..., description="Clearing prices")
    interactions: List[Interaction] = Field(
        ..., description="Smart contract interactions"
    )
    fulfillments: List[Fulfillment] = Field(
        default_factory=list, description="Order fulfillments"
    )
    fees: List[Fee] = Field(default_factory=list, description="Fees")
    jit_trades: List[JitTrade] = Field(default_factory=list, alias="jitTrades", description="JIT trades")
    liquidity_interactions: List[LiquidityInteraction] = Field(
        default_factory=list, alias="liquidityInteractions", description="Liquidity interactions"
    )
    custom_interactions: List[CustomInteraction] = Field(
        default_factory=list, alias="customInteractions", description="Custom interactions"
    )
    allowances: List[Allowance] = Field(
        default_factory=list, description="Token allowances"
    )

    model_config = {"populate_by_name": True}


class Solutions(BaseModel):
    """Collection of solutions."""

    solutions: List[Solution] = Field(..., description="List of solutions")

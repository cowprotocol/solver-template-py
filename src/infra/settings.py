"""
Configuration settings for the baseline solver.

This module provides configuration management using pydantic.
"""

from typing import List, Optional
from pydantic import BaseModel, Field, validator
from pydantic_settings import BaseSettings
from eth_typing import ChecksumAddress
from web3 import Web3


class SolverConfig(BaseSettings):
    """
    Baseline solver configuration.
    """

    # Network configuration
    chain_id: int = Field(
        default=1, description="Chain ID (1=mainnet, 100=gnosis)"  # Mainnet
    )

    # WETH address (changes per network)
    weth_address: ChecksumAddress = Field(
        default="0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",  # Mainnet WETH
        description="Wrapped ETH address for the network",
    )

    # Base tokens for pathfinding (liquid tokens for intermediary routes)
    base_tokens: List[ChecksumAddress] = Field(
        default=[
            "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",  # WETH
            "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48",  # USDC
            "0xdAC17F958D2ee523a2206206994597C13D831ec7",  # USDT
            "0x6B175474E89094C44Da98b954EedeAC495271d0F",  # DAI
            "0x2260FAC5E5542a773Aa44fBCfeDf7C193bc2C599",  # WBTC
        ],
        description="Tokens to consider as intermediary hops in pathfinding",
    )

    # Pathfinding configuration
    max_hops: int = Field(
        default=2,
        ge=0,
        le=3,
        description="Maximum hops in a trading path (0=direct, 1=one intermediary, 2=two intermediaries)",
    )

    # Partial fill configuration
    max_partial_attempts: int = Field(
        default=5,
        ge=1,
        le=10,
        description="Maximum attempts to solve a partially fillable order by halving amounts",
    )

    # Gas configuration
    solution_gas_offset: int = Field(
        default=50000,
        description="Gas units added to route estimate for settlement overhead",
    )

    # Price estimation
    native_token_price_estimation_amount: str = Field(
        default="1000000000000000000",  # 1 ETH in wei
        description="Amount of native token to use for price estimation",
    )

    # Optional Uniswap V3 support
    uni_v3_quoter_address: Optional[ChecksumAddress] = Field(
        default=None, description="Uniswap V3 Quoter V2 contract address (optional)"
    )

    # Gnosis Chain specific addresses (when chain_id=100)
    gnosis_weth_address: ChecksumAddress = Field(
        default="0xe91D153E0b41518A2Ce8Dd3D7944Fa863463a97d",  # WXDAI on Gnosis
        description="WETH address on Gnosis Chain",
    )

    gnosis_base_tokens: List[ChecksumAddress] = Field(
        default=[
            "0xe91D153E0b41518A2Ce8Dd3D7944Fa863463a97d",  # WXDAI
            "0x4ECaBa5870353805a9F068101A40E0f32ed605C6",  # USDT
            "0xDDAfbb505ad214D7b80b1f830fcCc89B60fb7A83",  # USDC
            "0x6A023CCd1ff6F2045C3309768eAd9E68F978f6e1",  # WETH on Gnosis
        ],
        description="Base tokens for Gnosis Chain",
    )

    @validator("weth_address", "gnosis_weth_address")
    def validate_weth_addresses(cls, v):
        if not Web3.is_address(v):
            raise ValueError(f"Invalid WETH address: {v}")
        return Web3.to_checksum_address(v)

    @validator("base_tokens", "gnosis_base_tokens")
    def validate_base_tokens(cls, v):
        validated_tokens = []
        for token in v:
            if not Web3.is_address(token):
                raise ValueError(f"Invalid base token address: {token}")
            validated_tokens.append(Web3.to_checksum_address(token))
        return validated_tokens

    @validator("uni_v3_quoter_address")
    def validate_quoter_address(cls, v):
        if v is not None and not Web3.is_address(v):
            raise ValueError(f"Invalid quoter address: {v}")
        return Web3.to_checksum_address(v) if v is not None else None

    class Config:
        env_prefix = "SOLVER_"
        env_file = ".env"

    def get_network_config(self):
        """Get configuration based on chain_id."""
        if self.chain_id == 100:  # Gnosis Chain
            return {
                "weth": self.gnosis_weth_address,
                "base_tokens": self.gnosis_base_tokens,
            }
        else:  # Mainnet or others
            return {"weth": self.weth_address, "base_tokens": self.base_tokens}


class Settings(BaseSettings):
    """Main application settings."""

    # API Configuration
    host: str = Field(default="0.0.0.0", env="HOST")
    port: int = Field(default=8080, env="PORT")

    # Solver Configuration
    default_solver_route: str = Field(
        default="baseline",
        pattern="^(baseline|mysolver)$",
        description="Default solver engine to use for /solve endpoint",
    )

    # Logging
    log_level: str = Field(
        default="INFO", pattern="^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$"
    )

    log_format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="Log format string",
    )

    # Include solver configuration
    solver: SolverConfig = Field(default_factory=SolverConfig)

    class Config:
        env_file = ".env"
        case_sensitive = False


# Global settings instance
settings = Settings()

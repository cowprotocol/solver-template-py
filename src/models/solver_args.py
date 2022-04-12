"""Argument parser for solve Request that combines query parameters with metadata"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from fastapi import Request

from src.util.schema import MetadataModel


@dataclass
class SolverArgs:
    """Parameters passed in POST URL"""

    auction_id: Optional[str]
    instance_name: str
    time_limit: int
    max_nr_exec_orders: int
    use_internal_buffers: bool
    use_external_prices: bool
    environment: Optional[str]
    gas_price: Optional[float]
    native_token: Optional[str]

    @classmethod
    def from_request(cls, request: Request, meta: MetadataModel) -> SolverArgs:
        """Parses Request query params dict as struct"""
        param_dict = request.query_params
        return cls(
            # Query Parameter Arguments
            instance_name=param_dict.get("instance_name", "Not Provided"),
            time_limit=int(param_dict.get("time_limit", 30)),
            max_nr_exec_orders=int(param_dict.get("max_nr_exec_orders", 100)),
            use_internal_buffers=bool(param_dict.get("use_internal_buffers", False)),
            use_external_prices=bool(param_dict.get("use_external_prices", False)),
            # Meta Data Arguments
            environment=meta.environment,
            gas_price=meta.gas_price,
            native_token=meta.native_token,
            # Both: Prioritize query params over metadata.
            auction_id=param_dict.get("auction_id", meta.auction_id),
        )

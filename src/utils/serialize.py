"""
Serialization utilities for CoW Protocol.

This module provides utilities for serializing and deserializing data.
"""

from typing import Any, Dict, List, Union
import json
from decimal import Decimal
from web3 import Web3
from hexbytes import HexBytes


def serialize_u256(value: int) -> str:
    """
    Serialize U256 value to string.

    Args:
        value: U256 value

    Returns:
        Serialized string
    """
    return str(value)


def deserialize_u256(value: str) -> int:
    """
    Deserialize string to U256 value.

    Args:
        value: Serialized string

    Returns:
        U256 value
    """
    if isinstance(value, str):
        if value.startswith("0x"):
            return int(value, 16)
        else:
            return int(value)
    return int(value)


def serialize_address(value: str) -> str:
    """
    Serialize address to string.

    Args:
        value: Address string

    Returns:
        Serialized address
    """
    return value.lower()


def serialize_hex(value: Union[int, bytes, str]) -> str:
    """
    Serialize value to hex string.

    Args:
        value: Value to serialize

    Returns:
        Hex string
    """
    return Web3.to_hex(value)


def deserialize_hex(value: str) -> bytes:
    """
    Deserialize hex string to bytes.

    Args:
        value: Hex string

    Returns:
        Bytes object
    """
    return HexBytes(value)


def serialize_decimal(value: Decimal) -> str:
    """
    Serialize Decimal to string.

    Args:
        value: Decimal value

    Returns:
        Serialized string
    """
    return str(value)


def deserialize_decimal(value: str) -> Decimal:
    """
    Deserialize string to Decimal.

    Args:
        value: Serialized string

    Returns:
        Decimal value
    """
    return Decimal(value)


def serialize_dict(data: Dict[str, Any]) -> str:
    """
    Serialize dictionary to JSON string.

    Args:
        data: Dictionary to serialize

    Returns:
        JSON string
    """
    return json.dumps(data, default=str)


def deserialize_dict(data: str) -> Dict[str, Any]:
    """
    Deserialize JSON string to dictionary.

    Args:
        data: JSON string

    Returns:
        Dictionary
    """
    return json.loads(data)


def serialize_list(data: List[Any]) -> str:
    """
    Serialize list to JSON string.

    Args:
        data: List to serialize

    Returns:
        JSON string
    """
    return json.dumps(data, default=str)


def deserialize_list(data: str) -> List[Any]:
    """
    Deserialize JSON string to list.

    Args:
        data: JSON string

    Returns:
        List
    """
    return json.loads(data)


def serialize_auction(auction: Dict[str, Any]) -> str:
    """
    Serialize auction data to JSON string.

    Args:
        auction: Auction data

    Returns:
        JSON string
    """
    return json.dumps(auction, default=str)


def deserialize_auction(data: str) -> Dict[str, Any]:
    """
    Deserialize JSON string to auction data.

    Args:
        data: JSON string

    Returns:
        Auction data dictionary
    """
    return json.loads(data)


def serialize_solution(solution: Dict[str, Any]) -> str:
    """
    Serialize solution data to JSON string.

    Args:
        solution: Solution data

    Returns:
        JSON string
    """
    return json.dumps(solution, default=str)


def deserialize_solution(data: str) -> Dict[str, Any]:
    """
    Deserialize JSON string to solution data.

    Args:
        data: JSON string

    Returns:
        Solution data dictionary
    """
    return json.loads(data)


def serialize_order(order: Dict[str, Any]) -> str:
    """
    Serialize order data to JSON string.

    Args:
        order: Order data

    Returns:
        JSON string
    """
    return json.dumps(order, default=str)


def deserialize_order(data: str) -> Dict[str, Any]:
    """
    Deserialize JSON string to order data.

    Args:
        data: JSON string

    Returns:
        Order data dictionary
    """
    return json.loads(data)


def serialize_liquidity(liquidity: Dict[str, Any]) -> str:
    """
    Serialize liquidity data to JSON string.

    Args:
        liquidity: Liquidity data

    Returns:
        JSON string
    """
    return json.dumps(liquidity, default=str)


def deserialize_liquidity(data: str) -> Dict[str, Any]:
    """
    Deserialize JSON string to liquidity data.

    Args:
        data: JSON string

    Returns:
        Liquidity data dictionary
    """
    return json.loads(data)


def serialize_token(token: Dict[str, Any]) -> str:
    """
    Serialize token data to JSON string.

    Args:
        token: Token data

    Returns:
        JSON string
    """
    return json.dumps(token, default=str)


def deserialize_token(data: str) -> Dict[str, Any]:
    """
    Deserialize JSON string to token data.

    Args:
        data: JSON string

    Returns:
        Token data dictionary
    """
    return json.loads(data)

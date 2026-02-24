"""
Fee conversion utilities for CoW Protocol solver.

This module provides centralized fee conversion functions to handle
different fee formats consistently across the solver.
"""

from typing import Union


def fee_to_basis_points(fee: Union[str, float, int]) -> int:
    """
    Convert fee to basis points (1/10000).

    Args:
        fee: Fee value in any supported format:
            - str: Decimal string (e.g., "0.003" -> 30)
            - float: Decimal number (e.g., 0.003 -> 30)
            - int: Already in basis points (e.g., 30 -> 30)

    Returns:
        int: Fee in basis points

    Examples:
        >>> fee_to_basis_points("0.003")
        30
        >>> fee_to_basis_points(0.003)
        30
        >>> fee_to_basis_points(30)
        30
    """
    if isinstance(fee, str):
        return int(float(fee) * 10000)
    elif isinstance(fee, float):
        return int(fee * 10000)
    return int(fee)


def basis_points_to_decimal(basis_points: int) -> float:
    """
    Convert basis points to decimal format.

    Args:
        basis_points: Fee in basis points (e.g., 30)

    Returns:
        float: Fee as decimal (e.g., 0.003)

    Examples:
        >>> basis_points_to_decimal(30)
        0.003
        >>> basis_points_to_decimal(25)
        0.0025
    """
    return basis_points / 10000.0

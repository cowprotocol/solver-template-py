"""
Mathematical utilities for CoW Protocol.

This module provides mathematical utilities with decimal precision.
"""

from decimal import Decimal, getcontext
from typing import Union
import math


# Set high precision for decimal arithmetic
getcontext().prec = 100


def to_decimal(value: Union[int, str, float, Decimal]) -> Decimal:
    """
    Convert value to Decimal with high precision.

    Args:
        value: Value to convert

    Returns:
        Decimal value
    """
    if isinstance(value, Decimal):
        return value
    return Decimal(str(value))


def from_decimal(value: Decimal) -> str:
    """
    Convert Decimal to string.

    Args:
        value: Decimal value

    Returns:
        String representation
    """
    return str(value)


def div_ceil(a: Union[int, Decimal], b: Union[int, Decimal]) -> int:
    """
    Ceiling division.

    Args:
        a: Dividend
        b: Divisor

    Returns:
        Ceiling of a / b
    """
    a = to_decimal(a)
    b = to_decimal(b)
    return int(math.ceil(a / b))


def div_floor(a: Union[int, Decimal], b: Union[int, Decimal]) -> int:
    """
    Floor division.

    Args:
        a: Dividend
        b: Divisor

    Returns:
        Floor of a / b
    """
    a = to_decimal(a)
    b = to_decimal(b)
    return int(math.floor(a / b))


def mul_div(
    a: Union[int, Decimal], b: Union[int, Decimal], c: Union[int, Decimal]
) -> int:
    """
    Multiply and divide with precision.

    Args:
        a: First factor
        b: Second factor
        c: Divisor

    Returns:
        (a * b) / c
    """
    a = to_decimal(a)
    b = to_decimal(b)
    c = to_decimal(c)
    return int((a * b) / c)


def mul_div_round(
    a: Union[int, Decimal], b: Union[int, Decimal], c: Union[int, Decimal]
) -> int:
    """
    Multiply and divide with rounding.

    Args:
        a: First factor
        b: Second factor
        c: Divisor

    Returns:
        Round((a * b) / c)
    """
    a = to_decimal(a)
    b = to_decimal(b)
    c = to_decimal(c)
    return int(round((a * b) / c))


def sqrt(value: Union[int, Decimal]) -> int:
    """
    Square root with precision.

    Args:
        value: Value to take square root of

    Returns:
        Square root as integer
    """
    value = to_decimal(value)
    return int(value.sqrt())


def pow(value: Union[int, Decimal], exp: Union[int, Decimal]) -> int:
    """
    Power operation with precision.

    Args:
        value: Base value
        exp: Exponent

    Returns:
        Power result as integer
    """
    value = to_decimal(value)
    exp = to_decimal(exp)
    return int(value**exp)


def log2(value: Union[int, Decimal]) -> int:
    """
    Base-2 logarithm.

    Args:
        value: Value to take log of

    Returns:
        Log2 result as integer
    """
    value = to_decimal(value)
    return int(math.log2(value))


def log10(value: Union[int, Decimal]) -> int:
    """
    Base-10 logarithm.

    Args:
        value: Value to take log of

    Returns:
        Log10 result as integer
    """
    value = to_decimal(value)
    return int(math.log10(value))


def min(a: Union[int, Decimal], b: Union[int, Decimal]) -> int:
    """
    Minimum of two values.

    Args:
        a: First value
        b: Second value

    Returns:
        Minimum value
    """
    a = to_decimal(a)
    b = to_decimal(b)
    return int(min(a, b))


def max(a: Union[int, Decimal], b: Union[int, Decimal]) -> int:
    """
    Maximum of two values.

    Args:
        a: First value
        b: Second value

    Returns:
        Maximum value
    """
    a = to_decimal(a)
    b = to_decimal(b)
    return int(max(a, b))


def clamp(
    value: Union[int, Decimal],
    min_val: Union[int, Decimal],
    max_val: Union[int, Decimal],
) -> int:
    """
    Clamp value between min and max.

    Args:
        value: Value to clamp
        min_val: Minimum value
        max_val: Maximum value

    Returns:
        Clamped value
    """
    value = to_decimal(value)
    min_val = to_decimal(min_val)
    max_val = to_decimal(max_val)
    return int(max(min_val, min(value, max_val)))


def abs(value: Union[int, Decimal]) -> int:
    """
    Absolute value.

    Args:
        value: Value to take absolute value of

    Returns:
        Absolute value
    """
    value = to_decimal(value)
    return int(abs(value))


def sign(value: Union[int, Decimal]) -> int:
    """
    Sign of value.

    Args:
        value: Value to get sign of

    Returns:
        -1, 0, or 1
    """
    value = to_decimal(value)
    if value > 0:
        return 1
    elif value < 0:
        return -1
    else:
        return 0


def is_zero(value: Union[int, Decimal]) -> bool:
    """
    Check if value is zero.

    Args:
        value: Value to check

    Returns:
        True if zero, False otherwise
    """
    value = to_decimal(value)
    return value == 0


def is_positive(value: Union[int, Decimal]) -> bool:
    """
    Check if value is positive.

    Args:
        value: Value to check

    Returns:
        True if positive, False otherwise
    """
    value = to_decimal(value)
    return value > 0


def is_negative(value: Union[int, Decimal]) -> bool:
    """
    Check if value is negative.

    Args:
        value: Value to check

    Returns:
        True if negative, False otherwise
    """
    value = to_decimal(value)
    return value < 0

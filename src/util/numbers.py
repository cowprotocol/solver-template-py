"""Utility methods for number handling"""
from decimal import Decimal


def decimal_to_str(number: Decimal) -> str:
    """Converts Decimal to string"""
    try:
        return f"{round(float(number), 12):.12f}".rstrip("0").rstrip(".")
    except ValueError as err:
        raise ValueError(f"Could not convert <{number}> into a string!") from err

"""Global constants."""

from decimal import Decimal


class Constants:
    """Configuration parameters for the solver."""

    # Precision of Decimal in strings (to be used as x.quantize(DECIMAL_STR_PREC)).
    DECIMAL_STR_PREC = Decimal("1e-10")

    # Should an exception be raised when the solution violates the
    # max sell amount constraint.
    RAISE_ON_MAX_SELL_AMOUNT_VIOLATION = False

    # Should an exception be raised when the solution violates the
    # limit exchange rate constraint.
    RAISE_ON_LIMIT_XRATE_VIOLATION = False

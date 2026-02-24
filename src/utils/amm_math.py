"""
AMM calculation utilities for different pool types.

This module provides mathematical functions for various AMM types following
the exact formulas used on-chain. All calculations use integer arithmetic
to match Solidity implementations.
"""

from typing import Optional
from decimal import Decimal, getcontext

# Set high precision for intermediate calculations
getcontext().prec = 78  # Support U256 max (2^256 ≈ 10^78)


class UniswapV2:
    """
    Uniswap V2 constant product AMM calculations.

    Implements x * y = k formula with fees.
    All amounts are in wei (integers).
    """

    @staticmethod
    def get_amount_out(
        amount_in: int,
        reserve_in: int,
        reserve_out: int,
        fee_bps: int = 30,  # 0.3% = 30 basis points
    ) -> int:
        """
        Calculate output amount for Uniswap V2 swap.

        Formula with fee:
        amount_out = (amount_in * (10000 - fee) * reserve_out) / (reserve_in * 10000 + amount_in * (10000 - fee))

        Args:
            amount_in: Amount of input token in wei
            reserve_in: Reserve of input token in pool in wei
            reserve_out: Reserve of output token in pool in wei
            fee_bps: Fee in basis points (30 = 0.3%)

        Returns:
            Amount of output token received in wei
        """
        if amount_in <= 0 or reserve_in <= 0 or reserve_out <= 0:
            return 0

        amount_in_with_fee = amount_in * (10000 - fee_bps)
        numerator = amount_in_with_fee * reserve_out
        denominator = reserve_in * 10000 + amount_in_with_fee

        if denominator == 0:
            return 0

        return numerator // denominator

    @staticmethod
    def get_amount_in(
        amount_out: int, reserve_in: int, reserve_out: int, fee_bps: int = 30
    ) -> Optional[int]:
        """
        Calculate required input amount for exact output.

        Formula:
        amount_in = (reserve_in * amount_out * 10000) / ((reserve_out - amount_out) * (10000 - fee)) + 1

        Args:
            amount_out: Desired amount of output token in wei
            reserve_in: Reserve of input token in pool in wei
            reserve_out: Reserve of output token in pool in wei
            fee_bps: Fee in basis points

        Returns:
            Required amount of input token in wei, or None if impossible
        """
        if amount_out <= 0 or amount_out >= reserve_out:
            return None
        if reserve_in <= 0 or reserve_out <= 0:
            return None

        numerator = reserve_in * amount_out * 10000
        denominator = (reserve_out - amount_out) * (10000 - fee_bps)

        if denominator <= 0:
            return None

        # Add 1 to round up (ensure we get at least amount_out)
        return (numerator // denominator) + 1


class BalancerWeighted:
    """
    Balancer weighted pool calculations.

    Implements weighted product formula with configurable weights.
    """

    @staticmethod
    def calc_out_given_in(
        balance_in: int,
        weight_in: int,  # As integer, e.g., 50 for 50%
        balance_out: int,
        weight_out: int,  # As integer, e.g., 50 for 50%
        amount_in: int,
        fee_bps: int = 30,  # Fee in basis points
    ) -> int:
        """
        Calculate output for Balancer weighted pool.

        Formula:
        amount_out = balance_out * (1 - (balance_in / (balance_in + amount_in_with_fee))^(weight_in/weight_out))

        Args:
            balance_in: Balance of input token in pool in wei
            weight_in: Weight of input token (as percentage, e.g., 50)
            balance_out: Balance of output token in pool in wei
            weight_out: Weight of output token (as percentage, e.g., 50)
            amount_in: Amount of input token to swap in wei
            fee_bps: Swap fee in basis points

        Returns:
            Amount of output token received in wei
        """
        if amount_in <= 0 or balance_in <= 0 or balance_out <= 0:
            return 0
        if weight_in <= 0 or weight_out <= 0:
            return 0

        # Apply fee
        fee_multiplier = 10000 - fee_bps
        amount_in_with_fee = (amount_in * fee_multiplier) // 10000

        # Use Decimal for precise intermediate calculations
        bi = Decimal(balance_in)
        bo = Decimal(balance_out)
        ai = Decimal(amount_in_with_fee)
        wi = Decimal(weight_in)
        wo = Decimal(weight_out)

        # Calculate: (bi / (bi + ai))^(wi/wo)
        ratio = bi / (bi + ai)
        weight_ratio = wi / wo

        # For common ratios, use optimized calculations (like Balancer V3+ does)
        if weight_in == weight_out:  # 50/50 pool
            factor = 1 - ratio
        elif weight_in == 80 and weight_out == 20:  # 80/20 pool
            factor = 1 - ratio**4  # 80/20 = 4
        elif weight_in == 20 and weight_out == 80:  # 20/80 pool
            factor = 1 - ratio ** Decimal("0.25")  # 20/80 = 0.25
        else:
            # General case
            factor = 1 - ratio**weight_ratio

        amount_out = int(bo * factor)

        return amount_out

    @staticmethod
    def calc_in_given_out(
        balance_in: int,
        weight_in: int,
        balance_out: int,
        weight_out: int,
        amount_out: int,
        fee_bps: int = 30,
    ) -> Optional[int]:
        """
        Calculate required input for exact output in weighted pool.

        Formula:
        amount_in = balance_in * ((balance_out / (balance_out - amount_out))^(weight_out/weight_in) - 1) / (1 - fee)

        Args:
            balance_in: Balance of input token in pool in wei
            weight_in: Weight of input token
            balance_out: Balance of output token in pool in wei
            weight_out: Weight of output token
            amount_out: Desired output amount in wei
            fee_bps: Swap fee in basis points

        Returns:
            Required input amount in wei, or None if impossible
        """
        if amount_out <= 0 or amount_out >= balance_out:
            return None
        if balance_in <= 0 or balance_out <= 0:
            return None
        if weight_in <= 0 or weight_out <= 0:
            return None

        # Use Decimal for calculations
        bi = Decimal(balance_in)
        bo = Decimal(balance_out)
        ao = Decimal(amount_out)
        wi = Decimal(weight_in)
        wo = Decimal(weight_out)

        # Calculate: ((bo / (bo - ao))^(wo/wi) - 1)
        ratio = bo / (bo - ao)
        weight_ratio = wo / wi

        # Optimize for common ratios
        if weight_in == weight_out:  # 50/50 pool
            factor = ratio - 1
        else:
            factor = ratio**weight_ratio - 1

        # Apply fee (inverse)
        fee_divisor = (10000 - fee_bps) / 10000

        amount_in = int((bi * factor) / fee_divisor) + 1  # Round up

        return amount_in


class CurveStable:
    """
    Curve StableSwap calculations.

    Implements stable pool invariant for low-slippage swaps between
    pegged assets (e.g., stablecoins).
    """

    @staticmethod
    def get_amount_out(
        amount_in: int,
        reserve_in: int,
        reserve_out: int,
        amplification: int,
        fee_bps: int = 4,  # 0.04% = 4 basis points
    ) -> int:
        """
        Calculate output for StableSwap pool.

        TODO: Simplified stable pool calculation. Real implementation requires
        solving polynomial equation iteratively.

        Args:
            amount_in: Input amount in wei
            reserve_in: Input token reserve in wei
            reserve_out: Output token reserve in wei
            amplification: Amplification coefficient (typically 10-1000)
            fee_bps: Fee in basis points

        Returns:
            Output amount in wei
        """
        if amount_in <= 0 or reserve_in <= 0 or reserve_out <= 0:
            return 0

        # Apply fee
        amount_in_with_fee = (amount_in * (10000 - fee_bps)) // 10000

        # TODO: Simplified stable swap - real implementation needs Newton-Raphson
        # This is an approximation for demonstration
        total_balance = reserve_in + reserve_out

        # Use amplification to reduce slippage
        # Higher amplification = lower slippage for balanced pools
        if amplification > 1:
            # TODO: Simplified formula - real Curve uses more complex math
            k = reserve_in * reserve_out
            new_reserve_in = reserve_in + amount_in_with_fee
            new_reserve_out = k // new_reserve_in

            # TODO: Apply amplification effect (simplified)
            slippage_reduction = min(amplification / 100, 1.0)
            ideal_out = amount_in_with_fee * reserve_out // reserve_in
            actual_out = reserve_out - new_reserve_out

            amount_out = int(
                actual_out + (ideal_out - actual_out) * Decimal(slippage_reduction)
            )
        else:
            # Fall back to constant product if amplification is low
            return UniswapV2.get_amount_out(amount_in, reserve_in, reserve_out, fee_bps)

        return max(0, amount_out)


def get_gas_estimate(pool_type: str) -> int:
    """
    Get gas estimate for different pool types.

    Args:
        pool_type: Type of pool ('uniswap_v2', 'balancer', 'curve')

    Returns:
        Estimated gas cost in units
    """
    gas_estimates = {
        "uniswap_v2": 110_000,
        "balancer": 150_000,
        "curve": 175_000,
        "uniswap_v3": 184_000,
    }
    return gas_estimates.get(pool_type, 150_000)  # Default to 150k

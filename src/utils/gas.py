"""Gas estimation utilities."""


class GasEstimator:
    """Estimates gas costs for settlement."""

    BASE_SETTLEMENT_GAS = 100_000
    PER_TRADE_GAS = 50_000
    PER_INTERACTION_GAS = 30_000

    def estimate_settlement_gas(self, trades: int, interactions: int) -> int:
        """Estimate total gas for settlement."""
        return (
            self.BASE_SETTLEMENT_GAS
            + trades * self.PER_TRADE_GAS
            + interactions * self.PER_INTERACTION_GAS
        )

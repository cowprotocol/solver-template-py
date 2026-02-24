"""
Dependency injection for solver engines.
"""

from src.engines.baseline.engine import BaselineEngine
from src.engines.mysolver.engine import MySolverEngine
from src.engines.base import SolverEngine
from src.infra.settings import settings


def get_baseline_engine() -> BaselineEngine:
    """Create baseline engine with configuration."""
    config = settings.solver.get_network_config()
    return BaselineEngine(
        weth_address=config["weth"],
        base_tokens=config["base_tokens"],
        max_hops=settings.solver.max_hops,
        max_partial_attempts=settings.solver.max_partial_attempts,
        solution_gas_offset=settings.solver.solution_gas_offset,
        native_token_price_estimation_amount=settings.solver.native_token_price_estimation_amount,
    )


def get_mysolver_engine() -> MySolverEngine:
    """Create custom solver engine."""
    return MySolverEngine()


# Legacy function names for backward compatibility
def baseline_engine() -> SolverEngine:
    """Get baseline solver engine instance."""
    return get_baseline_engine()


def mysolver_engine() -> SolverEngine:
    """Get MySolver engine instance."""
    return get_mysolver_engine()

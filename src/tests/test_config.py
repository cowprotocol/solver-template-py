"""
Test configuration system for the baseline solver.
"""

import pytest
from src.infra.settings import SolverConfig, Settings


def test_solver_config_defaults():
    """Test default solver configuration."""
    config = SolverConfig()

    assert config.chain_id == 1
    assert config.weth_address == "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2"
    assert len(config.base_tokens) >= 4
    assert config.max_hops == 2
    assert config.max_partial_attempts == 5
    assert config.solution_gas_offset == 50000
    assert config.native_token_price_estimation_amount == "1000000000000000000"


def test_gnosis_config():
    """Test Gnosis chain configuration."""
    config = SolverConfig(chain_id=100)
    network_config = config.get_network_config()

    assert network_config["weth"] == config.gnosis_weth_address
    assert len(network_config["base_tokens"]) >= 3


def test_mainnet_config():
    """Test mainnet configuration."""
    config = SolverConfig(chain_id=1)
    network_config = config.get_network_config()

    assert network_config["weth"] == config.weth_address
    assert len(network_config["base_tokens"]) >= 4


def test_engine_initialization():
    """Test baseline engine accepts configuration."""
    from src.engines.baseline.engine import BaselineEngine

    engine = BaselineEngine(
        weth_address="0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",
        base_tokens=["0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48"],
        max_hops=2,
    )

    assert engine.weth_address == "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2"
    assert len(engine.base_tokens) >= 2  # WETH + USDC
    assert engine.max_hops == 2


def test_settings_integration():
    """Test Settings class integration."""
    settings = Settings()

    assert settings.host == "0.0.0.0"
    assert settings.port == 8080
    assert settings.default_solver_route == "baseline"
    assert settings.log_level == "INFO"
    assert isinstance(settings.solver, SolverConfig)


def test_config_validation():
    """Test configuration validation."""
    # Test valid max_hops
    config = SolverConfig(max_hops=2)
    assert config.max_hops == 2

    # Test invalid max_hops (should raise validation error)
    with pytest.raises(ValueError):
        SolverConfig(max_hops=5)  # Should be <= 3

    # Test invalid max_partial_attempts
    with pytest.raises(ValueError):
        SolverConfig(max_partial_attempts=15)  # Should be <= 10


def test_network_config_switching():
    """Test network configuration switching."""
    # Mainnet
    config = SolverConfig(chain_id=1)
    network_config = config.get_network_config()
    assert network_config["weth"] == config.weth_address

    # Gnosis
    config = SolverConfig(chain_id=100)
    network_config = config.get_network_config()
    assert network_config["weth"] == config.gnosis_weth_address


def test_base_tokens_inclusion():
    """Test that WETH is included in base tokens."""
    from src.engines.baseline.engine import BaselineEngine

    engine = BaselineEngine(
        weth_address="0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",
        base_tokens=["0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48"],  # USDC only
        max_hops=2,
    )

    # WETH should be automatically added to base tokens
    assert "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2" in engine.base_tokens
    assert "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48" in engine.base_tokens


def test_dependency_injection():
    """Test dependency injection with configuration."""
    from src.infra.di import get_baseline_engine

    engine = get_baseline_engine()

    assert engine.weth_address is not None
    assert len(engine.base_tokens) > 0
    assert engine.max_hops >= 0
    assert engine.max_partial_attempts > 0
    assert engine.solution_gas_offset > 0

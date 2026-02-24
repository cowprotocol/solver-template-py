> [!WARNING]  
> This repository is outdated and does not currently function as a basis for implementing a solver.

# CoW Protocol Solver Template (Python)

A Python template for implementing CoW Protocol solvers. This template provides a basic structure and examples for building solvers that can participate in CoW Protocol auctions.

## Quick Start

🚀 **New to CoW Protocol solvers?** Start with our [Quick Start Guide](QUICKSTART.md) to get a complete solver stack running against the Barn API.

The Quick Start guide covers:
- Setting up the complete solver infrastructure (Autopilot + Driver + Python Solver)
- Connecting to CoW Protocol's staging environment
- Running your first solver against live auction data

## Overview

This template provides a modular, production-ready foundation for building CoW Protocol solvers with:

- **Multiple Engine Support**: Baseline and custom solver implementations
- **FastAPI Integration**: RESTful API with health checks and metrics
- **Modular Architecture**: Clean separation of concerns
- **Comprehensive Testing**: Full test suite with configuration validation
- **Development Tools**: Makefile with common development commands

## Prerequisites

- **Python 3.11+** — [Installation guide](https://www.python.org/downloads/)
- **Poetry** — [Installation guide](https://python-poetry.org/docs/#installation) (Python dependency management)
- **Rust v1.60.0+** — [Installation guide](https://www.rust-lang.org/tools/install) (for connecting to the driver)

## Installation

```bash
# Clone the repository
git clone https://github.com/cowprotocol/solver-template-py.git
cd solver-template-py

# Install dependencies
poetry install

# Verify installation
poetry run python -c "from src.infra.settings import settings; print('✅ Config OK:', settings.solver.chain_id)"
```

## Usage

### Start the Solver Server

```bash
# Using Makefile (Recommended)
make run

# Or manually
poetry run python -m src.infra.cli run
```

The solver will start on `http://localhost:8080` with the following endpoints:
- `GET /` - Root endpoint with service information
- `GET /healthz` - Health check
- `GET /metrics` - Prometheus metrics
- `POST /baseline/solve` - Baseline solver engine
- `POST /mysolver/solve` - MySolver engine (stub implementation)
- `POST /solve` - Default solver (configurable via DEFAULT_SOLVER_ROUTE)

### Test the Solver

```bash
# Health check
curl http://localhost:8080/healthz

# Test baseline solver
curl -X POST "http://127.0.0.1:8080/baseline/solve" \
  -H "Content-Type: application/json" \
  --data "@data/small_example.json"

# Test custom solver
curl -X POST "http://127.0.0.1:8080/mysolver/solve" \
  -H "Content-Type: application/json" \
  --data "@data/small_example.json"
```

## Development

### Available Commands

```bash
# Show all available commands
make help

# Start the solver server
make run

# Format code with black
make format

# Run all tests
make test

# Install dependencies
make install

# Clean up temporary files
make clean
```

### Manual Commands

```bash
# Start server
poetry run python -m src.infra.cli run --host 0.0.0.0 --port 8080

# Format code
poetry run black src/ --line-length 88

# Run tests
poetry run pytest src/tests/ -v

# Show configuration
poetry run python -m src.infra.cli config
```

### Testing

```bash
# Run all tests
poetry run pytest src/tests/

# Run specific test categories
poetry run pytest src/tests/test_config.py -v
poetry run pytest src/tests/test_api_health_metrics.py -v

# Run with coverage
poetry run pytest src/tests/ --cov=src
```

## Architecture

The solver follows a clean, modular architecture:

```
src/
├── domain/          # Business models (auction, order, solution, liquidity)
├── engines/         # Solver implementations (baseline, custom)
├── api/            # FastAPI endpoints and routing
├── infra/          # Configuration, logging, metrics
├── utils/          # Shared utilities (math, serialization)
└── tests/          # Test suite
```

### Key Components

- **API Layer**: FastAPI application with separate routers for each engine
- **Domain Layer**: Business models and logic (auction, solution, order, etc.)
- **Engine Layer**: Pluggable solver implementations (baseline, custom)
- **Infrastructure Layer**: Configuration, logging, metrics, and dependency injection
- **Utils Layer**: Shared utilities for math, serialization, and data conversion

## Implementation Guide

1. **Understand the Models**: Start by examining the domain models in `src/domain/`
2. **Study Engine Architecture**: Check `src/engines/base.py` for the engine protocol
3. **Implement Custom Solver**: Modify `src/engines/mysolver/engine.py` to implement your solver logic
4. **Add Pathfinding**: Implement algorithms to find optimal trading paths
5. **Handle AMMs**: Add support for different AMM protocols
6. **Optimize**: Implement price optimization and MEV protection
7. **Test**: Use the test suite in `src/tests/` to validate your implementation

## Schema Compatibility

This template uses the schema:
- **Input**: `Auction` model with proper field aliases
- **Output**: `Solutions` model matching the protocol specification
- **Field Names**: Uses Python snake_case with automatic camelCase JSON conversion

## Non-Production Ready Parts

This template contains areas marked with `TODO:` comments that indicate non-production ready implementations. These are simplified or stub implementations that need to be replaced with proper production code.

## Integration with CoW Protocol

For complete integration with the CoW Protocol ecosystem, see our [Quick Start Guide](QUICKSTART.md) which covers:

- Setting up the complete solver
- Connecting to the Barn API (staging environment)
- Running the autopilot and driver components
- Testing with live auction data

## References

- [CoW Protocol Solvers Tutorial](https://docs.cow.fi/cow-protocol/tutorials/solvers)
- [Settlement Contract](https://github.com/cowprotocol/contracts/blob/ff6fb7cad7787b8d43a6468809cacb799601a10e/src/contracts/GPv2Settlement.sol#L121-L143)
- [Interaction Model](https://github.com/cowprotocol/services/blob/cda5e36db34c55e7bf9eb4ea8b6e36ecb046f2b2/crates/shared/src/http_solver/model.rs#L125-L130)

## Contributing

Please feel free to submit issues and pull requests to improve the template for the community.
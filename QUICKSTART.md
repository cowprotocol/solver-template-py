# Quick Start: Running Python Solvers Against Barn API

This guide walks you through setting up and running Python-based solvers against CoW Protocol's staging environment (Barn API).

## What You'll Build

A complete solver stack running locally:
- **Autopilot**: Fetches live auctions from Barn API
- **Driver**: Routes auction data to your solver and manages solution submission
- **Python Solver**: Your custom solving logic

## Prerequisites

Before starting, ensure you have:

- **Rust & Cargo** — [Installation guide](https://www.rust-lang.org/tools/install)
- **Python 3.11+** — [Installation guide](https://www.python.org/downloads/)
- **Poetry** — [Installation guide](https://python-poetry.org/docs/#installation) (Python dependency management)
- **PostgreSQL** — Running instance (setup instructions in [services repo](https://github.com/cowprotocol/services?tab=readme-ov-file#postgres))
- **Ethereum RPC endpoint** — Required for blockchain access
  - Get a public endpoint from [Chainlist](https://chainlist.org/) (rate-limited)
  - **Alternative**: Use the [services playground setup](https://github.com/cowprotocol/services#playground) for a fully local environment

## Architecture Overview

```mermaid
sequenceDiagram
    box cowprotocol/services - Rust
        participant autopilot
        participant driver
    end
    box solver
        participant solver engine
    end
    autopilot->>driver: auction
    driver->>solver engine: auction
    solver engine->>driver: set of solutions
    driver->>autopilot: the best solution or quote,<br> depending on the request
    autopilot->>driver: request to publish a settlement,<br>in case this solver won
    driver->>driver: encode and publish the settlement
    autopilot->>autopilot: detect when the settlement is published<br>by monitoring the blockchain
```

## Setup Steps

### 1. Clone Required Repositories

```bash
# Clone the CoW Protocol services (contains autopilot & driver)
git clone https://github.com/cowprotocol/services.git

# Clone the Python solver template (in a separate directory)
git clone https://github.com/cowprotocol/solver-template-py.git
```

### 2. Install Python Solver Dependencies

Navigate to the solver template directory and install dependencies:

```bash
cd solver-template-py

# Install dependencies with Poetry
poetry install

# Verify installation
poetry run python -c "from src.infra.settings import settings; print('✅ Config OK:', settings.solver.chain_id)"
```

### 3. Test the Solver Setup

```bash
# Test configuration system
poetry run python -m pytest src/tests/test_config.py -v

# Show current configuration
poetry run python -m src.infra.cli config
```

### 4. Start Your Python Solver

The solver supports multiple engines in a single FastAPI application:

```bash
# Using Makefile (Recommended)
make run

# Or manually
poetry run python -m src.infra.cli run

# Alternative methods
poetry run python -m src._server
poetry run uvicorn src.api.app:app --host 0.0.0.0 --port 8080
```

The solver will start on `http://localhost:8080` with the following endpoints:
- `GET /` - Root endpoint with service information
- `GET /healthz` - Health check
- `GET /metrics` - Prometheus metrics
- `POST /baseline/solve` - Baseline solver engine
- `POST /mysolver/solve` - MySolver engine (stub implementation)
- `POST /solve` - Default solver (configurable via DEFAULT_SOLVER_ROUTE)

### 5. Test the Solver

```bash
# Health check
curl http://localhost:8080/healthz

# Test baseline solver
curl -X POST "http://127.0.0.1:8080/baseline/solve" \
  -H "accept: application/json" \
  -H "Content-Type: application/json" \
  --data "@data/small_example.json"

# Test custom solver
curl -X POST "http://127.0.0.1:8080/mysolver/solve" \
  -H "accept: application/json" \
  -H "Content-Type: application/json" \
  --data "@data/small_example.json"
```

### 6. PostgreSQL Setup

The autopilot requires a PostgreSQL database to track auction state.  
Follow the [database setup instructions](https://github.com/cowprotocol/services?tab=readme-ov-file#postgres) in the services repository README.

### 7. Configure the Driver

The driver (services repository) configuration file tells the system where to find your solver and how to interact with it.
Copy the example configuration from the Python solver template located at `examples/driver.toml` to your `services/crates/driver/` directory as `driver.toml`.

#### Key Configuration Changes from Default

The main modifications for Barn API setup:

- `orderbook-url`: Changed to `https://barn.api.cow.fi/mainnet/api` (staging environment)
- `[[solver]]` blocks: Added Python solver endpoints pointing to `localhost:8080` (FastAPI default port)
- `base-tokens`: Expanded list for better liquidity coverage
- Test private key: Using a known test key (never use for production!)

### 8. Start the Driver

In a new terminal, from the `services` directory:

```bash
cargo run --bin driver -- \
  --ethrpc "YOUR_RPC_ENDPOINT_HERE" \
  --config driver.toml
```

Replace `YOUR_RPC_ENDPOINT_HERE` with your Ethereum RPC URL.

Example:

```bash
cargo run --bin driver -- \
  --ethrpc "https://eth.llamarpc.com" \
  --config driver.toml
```

### 9. Start the Autopilot

In another terminal, from the `services` directory:

```bash
cargo run --bin autopilot -- \
  --node-url "YOUR_RPC_ENDPOINT_HERE" \
  --db-url "postgresql://postgres:password@localhost:5432/autopilot" \
  --drivers "baseline-python|http://localhost:11088/baseline-python|0x0000000000000000000000000000000000000000|0.1" \
  --drivers "python-solver|http://localhost:11088/python-solver|0x0000000000000000000000000000000000000000|0.1" \
  --native-price-estimators "baseline-python|http://localhost:11088/baseline-python" \
  --shadow "https://barn.api.cow.fi/mainnet/"
```

**Command Breakdown:**

- `--node-url`: Your Ethereum RPC endpoint  
- `--db-url`: PostgreSQL connection string  
- `--drivers`: Solver configurations in format `name|endpoint|address|priority`
  - The endpoint uses port `11088` (driver's internal port)  
  - Address `0x0000...` means using the account from driver config  
  - Priority `0.1` is a weight factor for solver selection  
- `--native-price-estimators`: Required parameter, points to a solver for price estimation  
- `--shadow`: Barn API URL for fetching live auction data

Optional: To also run the Rust baseline solver for comparison:

```bash
# Add this line to the autopilot command above
--drivers "baseline|http://localhost:11088/baseline|0x0000000000000000000000000000000000000000|0.1"
```

## Verifying Your Setup

### Check Solver Health

```bash
# Test baseline endpoint
curl http://127.0.0.1:8080/baseline/health

# Test custom solver endpoint
curl http://127.0.0.1:8080/mysolver/health
```

### Understanding the Flow

1. Autopilot fetches new auctions from Barn API periodically.  
2. Driver receives auction data and forwards it to all configured solvers.  
3. Your Python solver receives the auction, computes a solution, returns it.  
4. Driver evaluates all solutions and selects the best one.  
5. Autopilot manages solution submission (in staging, solutions aren't actually submitted on-chain).

## Place an Order

Navigate to [barn.cow.fi/](https://barn.cow.fi/) and place a tiny (real) order. You should see your driver pick it up and include it in the next auction being sent to your solver.

## Next Steps

### Development Workflow

- Modify and create your solver logic in `src/engines/mysolver/`
- Use `make format` to format your code with Black
- Run `make test` to execute the test suite

### Advanced Topics

- Multiple solvers: Add more `[[solver]]` blocks to test different strategies.  
- Custom liquidity sources: Modify the `[liquidity]` section in `driver.toml`.  
- Production deployment: See Orderbook API documentation for production endpoints.
- Rust baseline solver: Full implementation available in the services repository.

### Getting Help

- Check the [CoW Protocol documentation](https://docs.cow.fi/)
- Review the [services repository](https://github.com/cowprotocol/services) for additional context
- Join the [CoW Protocol Discord](https://discord.gg/cowprotocol) for community support

---

**Ready to start developing?** Check out the [main README](README.md) for detailed information about the solver template architecture and development workflow.
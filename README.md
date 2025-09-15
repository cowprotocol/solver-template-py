> [!WARNING]  
> This repository is outdated and does not currently function as a basis for implementing a solver.

# CoW Protocol Solver Template (Python)

A Python template for implementing CoW Protocol solvers. This template provides a basic structure and examples for building solvers that can participate in CoW Protocol auctions.

## Setup Project

Clone this repository

```sh
git clone git@github.com:cowprotocol/solver-template-py.git
cd solver-template-py
```

## Install Requirements

1. Python 3.11+ (tested with Python 3.11.9)
2. Rust v1.60.0+ (for connecting to the driver)

```sh
# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Run Solver Server

```shell
source venv/bin/activate
python -m src._server
```

The solver will start on `http://localhost:8000`

## Test the Solver

### Health Check
```shell
curl http://localhost:8000/health
```

### Solve an Auction
```shell
curl -X POST "http://127.0.0.1:8000/solve" \
  -H "accept: application/json" \
  -H "Content-Type: application/json" \
  --data "@data/small_example.json"
```

## Connect to the Orderbook

To connect your solver to the CoW Protocol orderbook, you need to run the driver (auction dispatcher) in DryRun mode. This will read from the staging environment on Gnosis Chain.

### Prerequisites
1. Clone the services repository:
```shell
git clone https://github.com/cowprotocol/services.git
cd services
```

2. Make sure your solver is running on `http://127.0.0.1:8000`

### Run the Driver
```shell
cargo run -p driver -- \
    --orderbook-url https://barn.api.cow.fi/xdai/api \
    --base-tokens 0xDDAfbb505ad214D7b80b1f830fcCc89B60fb7A83 \
    --node-url "https://rpc.gnosischain.com" \
    --cow-dex-ag-solver-url "http://127.0.0.1:8000" \
    --solver-account 0x7942a2b3540d1ec40b2740896f87aecb2a588731 \
    --solvers CowDexAg \
    --transaction-strategy DryRun \
    --log-filter=info,solver=debug
```

The driver will connect to the staging orderbook on Gnosis Chain (very low traffic) so you can work with your own orders.

## Place an Order

Navigate to [barn.cow.fi/](https://barn.cow.fi/) and place a tiny (real) order. You should see your driver pick it up and include it in the next auction being sent to your solver.

> **Note**: The driver and barn integration steps above have not been tested yet. They are included based on the original documentation and should be verified before use.

## Project Structure

```
src/
├── _server.py              # Main server entry point
├── models/                 # Data models
│   ├── batch_auction.py   # Auction data structures
│   ├── order.py           # Order data structures
│   ├── solver_args.py     # Solver configuration
│   └── ...
├── util/                   # Utility functions
│   ├── constants.py       # Protocol constants
│   ├── enums.py          # Enumerations
│   └── ...
└── ...

data/
├── small_example.json     # Example auction data
└── large_example.json     # Larger example auction data
```

## Implementation Guide

1. **Understand the Models**: Start by examining the data models in `src/models/`
2. **Implement Solver Logic**: Modify the solve endpoint in `src/_server.py`
3. **Add Pathfinding**: Implement algorithms to find optimal trading paths
4. **Handle AMMs**: Add support for different AMM protocols
5. **Optimize**: Implement price optimization and MEV protection

## References

- [CoW Protocol Solvers Tutorial](https://docs.cow.fi/cow-protocol/tutorials/solvers)
- [Settlement Contract](https://github.com/cowprotocol/contracts/blob/ff6fb7cad7787b8d43a6468809cacb799601a10e/src/contracts/GPv2Settlement.sol#L121-L143)
- [Interaction Model](https://github.com/cowprotocol/services/blob/cda5e36db34c55e7bf9eb4ea8b6e36ecb046f2b2/crates/shared/src/http_solver/model.rs#L125-L130)

## Contributing

This template is actively maintained. Please feel free to submit issues and pull requests to improve the template for the community.

## License

See LICENSE file for details.
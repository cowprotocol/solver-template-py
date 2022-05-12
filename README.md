# Setup Project

Clone this repository

```sh
git clone git@github.com:cowprotocol/solver-template-py.git
```

## Install Requirements

1. Python 3.10 (or probably also 3.9)
2. Rust v1.60.0 or Docker

```sh
python3 -m venv venv
source ./venv/bin/activate
pip install -r requirements.txt
```

# Run Solver Server

```shell
python -m src._server
```

This can also be run via docker with

```sh
docker run -p 8000:8000 gchr.io/cowprotocol/solver-template-py
```

# Feed an Auction Instance to the Solver

```shell
curl -X POST "http://127.0.0.1:8000/solve" \
  -H  "accept: application/json" \
  -H  "Content-Type: application/json" \
  --data "@data/small_example.json"
```

# Connect to our orderbook:

Run the driver (auction dispatcher in DryRun mode). Configured to read the orderbook
from our staging environment on Gnosis Chain. These parameters can be altered
in [.env](.env)

## With Docker

If you have docker installed then you can run this.

```shell
docker run -it --rm --env-file .env --add-host host.docker.internal:host-gateway ghcr.io/cowprotocol/services solver
```

or without an env file (as described in
the [How to Write a Solver Tutorial](https://docs.cow.fi/tutorials/how-to-write-a-solver))

```shell
docker run -it --rm --add-host host.docker.internal:host-gateway ghcr.io/cowprotocol/services solver \
--orderbook-url https://protocol-xdai.dev.gnosisdev.com \
--base-tokens 0xDDAfbb505ad214D7b80b1f830fcCc89B60fb7A83 \
--node-url "https://rpc.xdaichain.com" \
--cow-dex-ag-solver-url "http://127.0.0.1:8000" \
--solver-account 0x7942a2b3540d1ec40b2740896f87aecb2a588731 \
--solvers CowDexAg \
--transaction-strategy DryRun
```

## Without Docker

Clone the services project with

```shell
git clone https://github.com/cowprotocol/services.git
```

```shell
cargo run -p solver -- \
    --orderbook-url https://protocol-xdai.dev.gnosisdev.com \
    --base-tokens 0xDDAfbb505ad214D7b80b1f830fcCc89B60fb7A83 \
    --node-url "https://rpc.xdaichain.com" \
    --cow-dex-ag-solver-url "http://127.0.0.1:8000" \
    --solver-account 0x7942a2b3540d1ec40b2740896f87aecb2a588731 \
    --solvers CowDexAg \
    --transaction-strategy DryRun \
    --log-filter=info,solver=debug
```

# Place an order

Navigate to [barn.cowswap.exchange/](https://barn.cowswap.exchange/#/swap) and place a
tiny (real) order. See your driver pick it up and include it in the next auction being
sent to your solver

# References

- How to Build a Solver: https://docs.cow.fi/tutorials/how-to-write-a-solver
- In Depth Solver
  Specification: https://docs.cow.fi/off-chain-services/in-depth-solver-specification
- Settlement Contract (namely the settle
  method): https://github.com/cowprotocol/contracts/blob/ff6fb7cad7787b8d43a6468809cacb799601a10e/src/contracts/GPv2Settlement.sol#L121-L143
- Interaction Model (Currently missing from this framework): https://github.com/cowprotocol/services/blob/cda5e36db34c55e7bf9eb4ea8b6e36ecb046f2b2/crates/shared/src/http_solver/model.rs#L125-L130

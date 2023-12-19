# Setup Project

Clone this repository

```sh
git clone git@github.com:cowprotocol/solver-template-py.git
```

## Install Requirements

1. Python 3.11 or Docker (for running the solver)
2. Rust v1.60.0 or Docker (for running the autopilot and driver) REMOVE DRIVER FROM THIS TEMPLATE. THIS SHOULD BE EXPLAINED IN SOME TUTORIAL and not be part of this solver template

```sh
python3 -m venv venv
source ./venv/bin/activate
pip install -r requirements.txt
```

# Run Solver Server

```sh
python -m src._server
```

This can also be run via docker with

```sh
docker run -p 8000:8000 gchr.io/cowprotocol/solver-template-py
```

or build your own docker image with

```sh
docker build -t solver-template-py .
```

and run it with
```sh
docker run -p 8000:8000 solver-template-py
```

# Feed an Auction Instance to the Solver

```sh
curl -X POST "http://127.0.0.1:8000/solve" \
  -H  "accept: application/json" \
  -H  "Content-Type: application/json" \
  --data "@data/small_example.json"
```

# Connect to the orderbook TBD:

# References TBD

- How to Build a Solver: https://docs.cow.fi/tutorials/how-to-write-a-solver
- In Depth Solver
  Specification: https://docs.cow.fi/off-chain-services/in-depth-solver-specification

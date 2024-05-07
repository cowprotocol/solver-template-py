"""
This is the project's Entry point.
"""
from __future__ import annotations

import argparse
import decimal
import logging

import uvicorn
from dotenv import load_dotenv
from src.util.numbers import decimal_to_str
from fastapi import FastAPI, Request
from fastapi.middleware.gzip import GZipMiddleware
from pydantic import BaseSettings

from src.models.batch_auction import BatchAuction
from src.models.solver_args import SolverArgs
from src.util.schema import (
    BatchAuctionModel,
    SettledBatchAuctionModel,
)

# Set decimal precision.
decimal.getcontext().prec = 100

# Holds parameters passed on the command line when invoking the server.
# These will be merged with request solver parameters
SERVER_ARGS = None


# ++++ Interface definition ++++


# Server settings: Can be overridden by passing them as env vars or in a .env file.
# Example: PORT=8001 python -m src._server
class ServerSettings(BaseSettings):
    """Basic Server Settings"""

    host: str = "0.0.0.0"
    port: int = 8000


server_settings = ServerSettings()

# ++++ Endpoints: ++++


app = FastAPI(title="Batch auction solver")
app.add_middleware(GZipMiddleware)


@app.get("/health", status_code=200)
def health() -> bool:
    """Convenience endpoint to check if server is alive."""
    return True

async def fetch_best_rates(batch: BatchAuction):
    url = "http://central-server-url/bestRates"
    payload = {
        "sellTokenAddress": [order.sell_token for order in batch.orders],
        "buyTokenAddress": [order.buy_token for order in batch.orders],
        "sellTokenAmount": [order.sell_amount for order in batch.orders],
        "user": "optional_user_address"  # if needed
    }
    response = requests.post(url, json=payload)
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception("Failed to fetch best rates")

def generate_solution(batch: BatchAuction):
    return {
        "ref_token": batch.ref_token.value,
        "orders": {order.order_id: order.as_dict() for order in batch.orders if order.is_executed()},
        "prices": {str(key): decimal_to_str(value) for key, value in batch.prices.items()},
        "amms": {},
        "prices": {},
        "approvals": [],
        "interaction_data": [],
        "score": "0",
    }

@app.post("/notify", response_model=bool)
async def notify(request: Request) -> bool:
    """Print response from notify endpoint."""
    print(f"Notify request {await request.json()}")
    return True


@app.post("/solve", response_model=SettledBatchAuctionModel)
async def solve(problem: BatchAuctionModel, request: Request):  # type: ignore
    """API POST solve endpoint handler"""
    logging.debug(f"Received solve request {await request.json()}")
    solver_args = SolverArgs.from_request(request=request, meta=problem.metadata)

    batch = BatchAuction.from_dict(problem.dict(), solver_args.instance_name)

    # Fetch best rates for each token pair involved in the auction
    best_rates = await fetch_best_rates(batch)

    # Update batch auction with the fetched rates
    update_batch_with_best_rates(batch, best_rates)

    print("Received Batch Auction", batch.name)
    print("Parameters Supplied", solver_args)

    # 1. Solve BatchAuction: update batch_auction with
    batch.solve()
    print("in solve",99)

    trivial_solution = {
        "ref_token": batch.ref_token.value,
        "orders": {order.order_id: order.as_dict() for order in batch.orders if order.is_executed() },
        "prices": {str(key): decimal_to_str(value) for key, value in batch.prices.items()},
        "amms": {},
        "prices": {},
        "approvals": [],
        "interaction_data": [],
        "score": "0",
    }

    print("\n\n*************\n\nReturning solution: " + str(trivial_solution))
    return trivial_solution


# ++++ Server setup: ++++


if __name__ == "__main__":
    load_dotenv()

    parser = argparse.ArgumentParser(
        fromfile_prefix_chars="@",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # TODO - enable flag to write files to persistent storage
    # parser.add_argument(
    #     "--write_auxiliary_files",
    #     type=bool,
    #     default=False,
    #     help="Write auxiliary instance and optimization files, or not.",
    # )

    parser.add_argument(
        "--log-level",
        type=str,
        default="info",
        help="Log level",
    )

    SERVER_ARGS = parser.parse_args()
    uvicorn.run(
        "__main__:app",
        host=server_settings.host,
        port=server_settings.port,
        log_level=SERVER_ARGS.log_level,
    )

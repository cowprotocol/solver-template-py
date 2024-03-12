"""
This is the project's Entry point.
"""
from __future__ import annotations

import argparse
import decimal
import logging
from typing import Any
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.gzip import GZipMiddleware
from pydantic import BaseSettings

from src.models.solver_args import SolverArgs

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


@app.post("/notify", response_model=bool)
async def notify(request: Request) -> bool:
    """Print response from notify endpoint."""
    print(f"Notify request {await request.json()}")
    return True


@app.post("/solve")
async def solve(problem: dict, request: Request):  # type: ignore
    """API POST solve endpoint handler"""
    logging.debug(f"Received solve request {await request.json()}")
    #return
    trivial_solution = {
        "solutions": []
    }
    gas_price = int(problem['effectiveGasPrice'])
    orders = problem['orders']
    found_solution = False
    for o in orders:
        if o['sellToken'] == o['buyToken']:
            found_solution = True
            break
    if not found_solution:
        print("\n\n*************\n\nReturning solution: " + str(trivial_solution))
        return trivial_solution
    
    gas_cost = 150000 * gas_price
    token = o['sellToken']
    sell_amount = int(o['sellAmount'])
    buy_amount = int(o['buyAmount'])
    tokens = problem['tokens']
    native_price = int(tokens[token]['referencePrice'])
    fee_in_sell = int(gas_cost * 10**18 / native_price)
    if sell_amount - fee_in_sell > buy_amount:
        solution = {
            'solutions': [ 
                {
                    'id': 0,
                    'prices': {token: '1'},
                    'trades': [
                        {
                            'kind': 'fulfillment',
                            'order': o['uid'],
                            'fee': str(fee_in_sell),
                            'executedAmount': str(sell_amount - fee_in_sell)
                        }
                    ],
                    'interactions': [],
                    'score': {
                        'kind': 'solver',
                        'score': str((sell_amount - fee_in_sell - buy_amount) * native_price // 10**18)
                    }
                }
            ]
        }
        print("\n\n*************\n\nReturning solution: " + str(solution))
        return solution
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

"""
This is the project's Entry point.
"""
from __future__ import annotations

import argparse
import decimal
import logging

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, Request
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


@app.get("/health", status_code=200)
def health() -> bool:
    """Convenience endpoint to check if server is alive."""
    return True


@app.post("/solve", response_model=SettledBatchAuctionModel)
async def solve(problem: BatchAuctionModel, request: Request):  # type: ignore
    """API POST solve endpoint handler"""
    logging.debug(f"Received solve request {await request.json()}")
    solver_args = SolverArgs.from_request(request=request, meta=problem.metadata)

    batch = BatchAuction.from_dict(problem.dict(), solver_args.instance_name)

    print("Received Batch Auction", batch.name)
    print("Parameters Supplied", solver_args)

    # 1. Solve BatchAuction: update batch_auction with
    # batch.solve()

    sample_output = {
        "ref_token": batch.ref_token.value,
        "orders": {order.order_id: order.as_dict() for order in batch.orders},
        "prices": {
            "0x4e3fbd56cd56c3e72c1403e103b45db9da5b9d2b": 10658174560450550,
            "0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2": 1000000000000000000,
            "0xdac17f958d2ee523a2206206994597c13d831ec7": 314968423380884,
        },
        "amms": {
            "6": {
                "kind": "ConstantProduct",
                "reserves": {
                    "0x4e3fbd56cd56c3e72c1403e103b45db9da5b9d2b": "151047198637918194794625",
                    "0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2": "1615731604381456130940",
                },
                "fee": "0.003",
                "cost": {
                    "amount": "1668264347574664",
                    "token": "0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2",
                },
                "mandatory": False,
                "id": "6",
                "execution": [
                    {
                        "buy_token": "0x4e3fbd56cd56c3e72c1403e103b45db9da5b9d2b",
                        "exec_buy_amount": 4416092913242591886,
                        "sell_token": "0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2",
                        "exec_sell_amount": 47095265163205056,
                        "exec_sell_amount_nfee": 47236971948467672,
                        "liquidity_fee": 1.3248278739727776e16,
                        "exec_plan": {"position": 0, "sequence": 0},
                    }
                ],
            },
            "3": {
                "kind": "ConstantProduct",
                "reserves": {
                    "0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2": "10052659476364989570713",
                    "0xdac17f958d2ee523a2206206994597c13d831ec7": "31939939882438",
                },
                "fee": "0.003",
                "cost": {
                    "amount": "1668264347574664",
                    "token": "0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2",
                },
                "mandatory": False,
                "id": "3",
                "execution": [
                    {
                        "buy_token": "0xdac17f958d2ee523a2206206994597c13d831ec7",
                        "exec_buy_amount": 55272822,
                        "sell_token": "0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2",
                        "exec_sell_amount": 17344146155103392,
                        "exec_sell_amount_nfee": 17396335070270996,
                        "liquidity_fee": 165818.46600000001,
                        "exec_plan": {"position": 1, "sequence": 0},
                    }
                ],
            },
        },
    }
    return sample_output


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

"""
This is the project's Entry point.
"""

import logging
from typing import Any
import uvicorn
from fastapi import FastAPI

# from src.models.solve_model import Auction, Solution

logging.basicConfig(level=logging.DEBUG)

# ++++ Endpoints: ++++

app = FastAPI(title="Batch auction solver")


@app.post("/notify", response_model=bool)
async def notify(notification: dict[str, Any]) -> bool:
    """Print response from notify endpoint."""
    logging.debug(f"Notification: {notification}")
    return True


# @app.post("/solve", response_model=Solution)
# async def solve(auction: Auction, request: Request):  # type: ignore
@app.post("/solve")
async def solve(auction: dict[str, Any]) -> dict[str, Any]:
    """API POST solve endpoint handler"""
    logging.debug(f"Received solve request: {auction}")

    # 1. Solve Auction
    # (add code)

    solution = {
        "id": "123",
        "trades": [],
        "prices": {},
        "interactions": [],
        "solver": "solvertemplate",
        "score": "0",
        "weth": "0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2",
    }

    logging.debug(f"Returning solution: {solution}")

    # return Solution(**solution)
    return solution


# ++++ Server setup: ++++


if __name__ == "__main__":
    uvicorn.run(
        "__main__:app",
        host="0.0.0.0",
        port=8000,
    )

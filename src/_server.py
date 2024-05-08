"""
This is the project's Entry point.
"""
from __future__ import annotations

import argparse
import decimal
import logging
import requests
import asyncio # Ensure requests is imported
import httpx


import uvicorn
from dotenv import load_dotenv
from src.util.numbers import decimal_to_str
from fastapi import FastAPI, Request , HTTPException
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

# Example Token and Order class definitions
class Token:
    def __init__(self, address):
        self.address = address

class Order:
    def __init__(self, sell_token, buy_token, sell_amount, buy_amount):
        self.sell_token = sell_token
        self.buy_token = buy_token
        self.sell_amount = sell_amount
        self.buy_amount = buy_amount

class BatchAuction:
    def __init__(self, tokens, orders):
        self.tokens = tokens
        self.orders = orders

# Token instances
token_weth = Token("0x6a023ccd1ff6f2045c3309768ead9e68f978f6e1")
token_cow = Token("0x177127622c4a00f3d409b75571e12cb3c8973d3c")

# Order instances using the above tokens
order1 = Order(
    sell_token=token_weth,
    buy_token=token_cow,
    sell_amount="12000000000000000000",
    buy_amount="100000000000000000000"
)
order2 = Order(
    sell_token=token_cow,
    buy_token=token_weth,
    sell_amount="100000000000000000000",
    buy_amount="12000000000000000000"
)

# List of tokens and orders
tokens = [token_weth, token_cow]
orders = [order1, order2]

batch_auction = BatchAuction(tokens, orders)

def display_auction_details(batch):
    for order in batch.orders:
        print(f"Sell Token: {order.sell_token.address}, Buy Token: {order.buy_token.address}, Sell Amount: {order.sell_amount}, Buy Amount: {order.buy_amount}")

# Display details of the batch auction
display_auction_details(batch_auction)
# ++++ Endpoints: ++++





app = FastAPI(title="Batch auction solver")
app.add_middleware(GZipMiddleware)


class ServerSettings(BaseSettings):
    """Basic Server Settings"""
    host: str = "0.0.0.0"
    port: int = 8000
    central_server_url: str = "http://localhost:3000"

server_settings = ServerSettings()

@app.on_event("startup")
async def startup_event():
    await main()
    # """Event that runs at the startup of the FastAPI application."""
    # batch = create_batch_auction()
    # best_rates = await fetch_best_rates_from_central_server(batch)
    # if best_rates:
    #     update_batch_with_best_rates(batch, best_rates)
    # print("Startup Event Completed: Best rates fetched and batch updated.")

@app.get("/health", status_code=200)
def health() -> bool:
    """Convenience endpoint to check if server is alive."""
    return True

@app.post("/notify", response_model=bool)
async def notify(request: Request) -> bool:
    """Print response from notify endpoint."""
    print(f"Notify request {await request.json()}")
    return True

# Fetch best rates function
# async def fetch_best_rates_from_central_server(sell_tokens, buy_tokens, sell_amounts):
async def fetch_best_rates_from_central_server(sell_tokens, buy_tokens, sell_amounts, user_address):
    url = "http://localhost:3000/bestRates"
    payload = {
        "sellTokenAddress": [token.address for token in sell_tokens],
        "buyTokenAddress": [token.address for token in buy_tokens],
        "sellTokenAmount": sell_amounts,
        "user": user_address
    }
    async with httpx.AsyncClient() as client:
        response = await client.post(url, json=payload)
        if response.status_code == 200:
            return response.json()
        else:
            raise HTTPException(status_code=response.status_code, detail="Failed to fetch best rates from central server")
        
# # Function to update batch auction with fetched rates
def update_batch_with_best_rates(batch: BatchAuction, best_rates):
    # Example logic to update orders based on fetched rates
    for rate in best_rates:
        # Assuming each rate item contains 'order_id' and new 'sell_amount' and 'buy_amount'
        order = next((order for order in batch.orders if order.order_id == rate['order_id']), None)
        if order:
            order.update_rate(rate['new_sell_amount'], rate['new_buy_amount'])

# # Function to generate the solution
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

@app.post("/solve", response_model=SettledBatchAuctionModel)
async def solve(problem: BatchAuctionModel, request: Request):
    logging.debug(f"Received solve request {await request.json()}")
    solver_args = SolverArgs.from_request(request=request, meta=problem.metadata)
    batch = BatchAuction.from_dict(problem.dict(), solver_args.instance_name)

    # Fetch and apply best rates
    best_rates = await fetch_best_rates_from_central_server(
        [order.sell_token for order in batch.orders],
        [order.buy_token for order in batch.orders],
        [order.sell_amount for order in batch.orders]
    )
    if best_rates:
        update_batch_with_best_rates(batch, best_rates)

    batch.solve()
    trivial_solution = generate_solution(batch)
    print("\n\n*************\n\nReturning solution: " + str(trivial_solution))
    return trivial_solution

def create_batch_auction() -> BatchAuction:
    """Create a BatchAuction object with predefined tokens and orders."""
    token_weth = Token("0x6a023ccd1ff6f2045c3309768ead9e68f978f6e1")
    token_cow = Token("0x177127622c4a00f3d409b75571e12cb3c8973d3c")
    order1 = Order(token_weth, token_cow, "12000000000000000000", "100000000000000000000")
    order2 = Order(token_cow, token_weth, "100000000000000000000", "12000000000000000000")
    return BatchAuction([token_weth, token_cow], [order1, order2])


async def main():
    # Initialize tokens and orders
    token_weth = Token("0xbb4CdB9CBd36B01bD1cBaEBF2De08d9173bc095c")
    token_cow = Token("0xCC42724C6683B7E57334c4E856f4c9965ED682bD")
    order1 = Order(token_weth, token_cow, "1000000000000", "5000000000000")
    
    # Create a batch auction with these tokens and orders
    batch = BatchAuction([token_weth, token_cow], [order1])
    
    # User address, can also be configured via environment variables
    user_address = "0xE7C7f2E2D17FF5B1c0e291d5b3e37Fe56b858227"
    
    # Fetch rates
    best_rates = await fetch_best_rates_from_central_server(
        [order.sell_token for order in batch.orders], 
        [order.buy_token for order in batch.orders], 
        [order.sell_amount for order in batch.orders], 
        user_address
    )
    print("Best Rates:", best_rates)

if __name__ == "__main__":
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-level", type=str, default="info", help="Log level")
    SERVER_ARGS = parser.parse_args()
    uvicorn.run("__main__:app", host=server_settings.host, port=server_settings.port, log_level=SERVER_ARGS.log_level)
    # asyncio.run(main())


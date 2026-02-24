"""
FastAPI application setup for the CoW Protocol solver.
"""

from fastapi import FastAPI, Request, Response, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
import time
import logging

from .healthz import health_check
from ..infra.metrics import (
    record_request,
    record_solution,
    record_auction_orders,
    set_active_requests,
    get_metrics,
    get_metrics_content_type,
)
from .routers import baseline, mysolver
from src.domain.auction import Auction
from src.domain.solution import Solutions
from src.infra.settings import settings
from src.infra.di import baseline_engine, mysolver_engine

logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="CoW Protocol Solver",
    description="Multi-engine solver implementation for CoW Protocol auctions",
    version="0.0.1",
)

# Add middleware
app.add_middleware(GZipMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"],
)


@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    """Middleware to collect metrics."""
    start_time = time.time()

    # Increment active requests
    set_active_requests(1)

    try:
        response = await call_next(request)
        duration = time.time() - start_time

        # Record metrics
        record_request(
            method=request.method,
            endpoint=request.url.path,
            status=str(response.status_code),
            duration=duration,
        )

        return response

    except Exception as e:
        duration = time.time() - start_time

        # Record error metrics
        record_request(
            method=request.method,
            endpoint=request.url.path,
            status="500",
            duration=duration,
        )

        raise

    finally:
        # Decrement active requests
        set_active_requests(0)


# Include routers
app.include_router(baseline.router)
app.include_router(mysolver.router)


@app.get("/healthz")
async def health():
    """Health check endpoint."""
    return health_check()


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    return Response(content=get_metrics(), media_type=get_metrics_content_type())


@app.post("/solve", response_model=Solutions)
async def solve(auction: Auction):
    """
    Solve a CoW Protocol auction using the default solver.

    Args:
        auction: The auction to solve

    Returns:
        Solutions object containing the solver's solutions
    """
    try:
        logger.info(f"Received solve request for auction: {auction.id}")
        logger.info(f"Orders: {len(auction.orders)}, Tokens: {len(auction.tokens)}")

        # Record auction metrics
        record_auction_orders(len(auction.orders))

        # Get the default solver engine
        if settings.default_solver_route == "baseline":
            engine = baseline_engine()
            logger.info("Using baseline engine")
        elif settings.default_solver_route == "mysolver":
            engine = mysolver_engine()
            logger.info("Using mysolver engine")
        else:
            raise ValueError(
                f"Unknown default solver route: {settings.default_solver_route}"
            )

        # Solve the auction
        solutions = await engine.solve(auction)

        # Record solution metrics
        record_solution("success")
        logger.info(
            f"Solved auction successfully, returning {len(solutions.solutions)} solutions"
        )

        return solutions

    except Exception as e:
        logger.error(f"Error solving auction: {e}")
        import traceback

        logger.error(f"Traceback: {traceback.format_exc()}")
        record_solution("error")
        raise


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "CoW Protocol Solver",
        "version": "1.0.0",
        "status": "running",
        "engines": ["baseline", "mysolver"],
        "default_engine": settings.default_solver_route,
    }


@app.options("/")
async def root_options():
    """Root endpoint OPTIONS handler for CORS."""
    from fastapi import Response

    response = Response(content="OK")
    response.headers["access-control-allow-origin"] = "*"
    response.headers["access-control-allow-methods"] = "GET, POST, PUT, DELETE, OPTIONS"
    response.headers["access-control-allow-headers"] = "*"
    return response

"""
MySolver router.
"""

import logging
from fastapi import APIRouter, Depends
from src.domain.auction import Auction
from src.domain.solution import Solutions
from src.engines.mysolver.engine import MySolverEngine
from src.infra.di import mysolver_engine

router = APIRouter(prefix="/mysolver", tags=["mysolver"])
logger = logging.getLogger(__name__)


@router.post("/solve")
async def solve_mysolver(
    auction: Auction, engine: MySolverEngine = Depends(mysolver_engine)
) -> Solutions:
    """Solve auction using custom solver engine."""

    result = await engine.solve(auction)

    logger.info(f"MYSOLVER SOLVER - SOLUTION:")
    logger.info(f" -> AUCTION ID: {auction.id}")
    logger.info(f" -> SOLUTIONS: {len(result.solutions)}")
    logger.info(f"===================================")

    return result


@router.post("/notify")
async def notify_mysolver(
    notification: dict,
) -> dict:
    logger.info(f"MYSOLVER NOTIFY:\n {notification}")
    logger.info(f"===================================")
    # For now, just acknowledge the notification
    # In a real implementation, this might trigger something
    return {"status": "acknowledged"}

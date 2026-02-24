"""
Baseline solver router.
"""

import logging
from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from src.domain.auction import Auction
from src.domain.solution import Solutions
from src.engines.baseline.engine import BaselineEngine
from src.infra.di import baseline_engine

router = APIRouter(prefix="/baseline", tags=["baseline"])
logger = logging.getLogger(__name__)


@router.post("/solve")
async def solve_baseline(
    auction: Auction, engine: BaselineEngine = Depends(baseline_engine)
) -> Solutions:
    """Solve auction using baseline engine."""
    result = await engine.solve(auction)
    response_dict = result.model_dump(by_alias=True)
    return JSONResponse(content=response_dict)


@router.post("/notify")
async def notify_baseline(
    notification: dict,
) -> dict:
    """Handle solver notifications."""
    return {"status": "acknowledged"}

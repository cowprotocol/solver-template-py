"""
Validation utilities for auction data.
"""

import logging
from src.domain.auction import Auction

logger = logging.getLogger(__name__)


def validate_auction(auction: Auction) -> bool:
    """
    Validate auction data for all solver engines.

    This function provides common validation logic that can be shared
    across all solver engines to ensure consistent validation behavior.

    Args:
        auction: The auction to validate

    Returns:
        True if valid, False otherwise
    """
    # Basic validation
    if not auction.id:
        logger.warning("Auction has no ID")
        return False

    if not auction.orders:
        logger.warning("Auction has no orders")
        return False

    if not auction.tokens:
        logger.warning("Auction has no tokens")
        return False

    return True

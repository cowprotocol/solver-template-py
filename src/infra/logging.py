"""
Structured logging for the baseline solver.

This module provides structured logging configuration.
"""

import logging
import sys
from typing import Optional
from .settings import settings


def setup_logging(log_level: Optional[str] = None) -> None:
    """
    Setup structured logging.

    Args:
        log_level: Optional log level override
    """
    level = log_level or settings.log_level

    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=settings.log_format,
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # Configure specific loggers
    logging.getLogger("uvicorn").setLevel(logging.INFO)
    logging.getLogger("fastapi").setLevel(logging.INFO)
    logging.getLogger("prometheus_client").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance.

    Args:
        name: Logger name

    Returns:
        Logger instance
    """
    return logging.getLogger(name)


def log_request(method: str, endpoint: str, status: int, duration: float) -> None:
    """
    Log a request.

    Args:
        method: HTTP method
        endpoint: Endpoint path
        status: HTTP status code
        duration: Request duration in seconds
    """
    logger = get_logger("solver.requests")
    logger.info(
        f"Request: {method} {endpoint} - Status: {status} - Duration: {duration:.3f}s"
    )


def log_solution(
    auction_id: str, order_count: int, solution_count: int, duration: float
) -> None:
    """
    Log a solution generation.

    Args:
        auction_id: Auction identifier
        order_count: Number of orders
        solution_count: Number of solutions
        duration: Solution duration in seconds
    """
    logger = get_logger("solver.solutions")
    logger.info(
        f"Solution: Auction {auction_id} - Orders: {order_count} - Solutions: {solution_count} - Duration: {duration:.3f}s"
    )


def log_error(
    error_type: str, message: str, exception: Optional[Exception] = None
) -> None:
    """
    Log an error.

    Args:
        error_type: Error type
        message: Error message
        exception: Optional exception
    """
    logger = get_logger("solver.errors")
    if exception:
        logger.error(f"Error: {error_type} - {message}", exc_info=exception)
    else:
        logger.error(f"Error: {error_type} - {message}")


def log_performance(
    operation: str, duration: float, details: Optional[str] = None
) -> None:
    """
    Log a performance metric.

    Args:
        operation: Operation name
        duration: Duration in seconds
        details: Optional details
    """
    logger = get_logger("solver.performance")
    if details:
        logger.info(f"Performance: {operation} - Duration: {duration:.3f}s - {details}")
    else:
        logger.info(f"Performance: {operation} - Duration: {duration:.3f}s")


def log_auction(
    auction_id: str, order_count: int, token_count: int, liquidity_count: int
) -> None:
    """
    Log an auction.

    Args:
        auction_id: Auction identifier
        order_count: Number of orders
        token_count: Number of tokens
        liquidity_count: Number of liquidity sources
    """
    logger = get_logger("solver.auctions")
    logger.info(
        f"Auction: {auction_id} - Orders: {order_count} - Tokens: {token_count} - Liquidity: {liquidity_count}"
    )


def log_startup(service_name: str, version: str, host: str, port: int) -> None:
    """
    Log service startup.

    Args:
        service_name: Service name
        version: Service version
        host: Server host
        port: Server port
    """
    logger = get_logger("solver.startup")
    logger.info(f"Starting {service_name} v{version} on {host}:{port}")


def log_shutdown(service_name: str) -> None:
    """
    Log service shutdown.

    Args:
        service_name: Service name
    """
    logger = get_logger("solver.shutdown")
    logger.info(f"Shutting down {service_name}")


def log_health_check(status: str, details: Optional[str] = None) -> None:
    """
    Log a health check.

    Args:
        status: Health status
        details: Optional details
    """
    logger = get_logger("solver.health")
    if details:
        logger.info(f"Health check: {status} - {details}")
    else:
        logger.info(f"Health check: {status}")


def log_metrics(metric_name: str, value: float, labels: Optional[dict] = None) -> None:
    """
    Log a metric.

    Args:
        metric_name: Metric name
        value: Metric value
        labels: Optional labels
    """
    logger = get_logger("solver.metrics")
    if labels:
        logger.info(f"Metric: {metric_name} = {value} - Labels: {labels}")
    else:
        logger.info(f"Metric: {metric_name} = {value}")

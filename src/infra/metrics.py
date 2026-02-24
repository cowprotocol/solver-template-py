"""
Prometheus metrics collection for the baseline solver.

This module provides metrics collection using prometheus_client.
"""

from typing import Dict, Any
import time
from prometheus_client import (
    Counter,
    Histogram,
    Gauge,
    generate_latest,
    CONTENT_TYPE_LATEST,
)


# Metrics definitions
REQUEST_COUNT = Counter(
    "solver_requests_total",
    "Total number of solver requests",
    ["method", "endpoint", "status"],
)

REQUEST_DURATION = Histogram(
    "solver_request_duration_seconds",
    "Request duration in seconds",
    ["method", "endpoint"],
)

ACTIVE_REQUESTS = Gauge("solver_active_requests", "Number of active requests")

SOLVER_SOLUTIONS = Counter(
    "solver_solutions_total", "Total number of solutions generated", ["status"]
)

SOLVER_ORDERS = Histogram(
    "solver_orders_per_auction",
    "Number of orders per auction",
    buckets=[1, 5, 10, 25, 50, 100, 250, 500, 1000],
)

SOLVER_TOKENS = Histogram(
    "solver_tokens_per_auction",
    "Number of tokens per auction",
    buckets=[1, 5, 10, 25, 50, 100, 250, 500, 1000],
)

SOLVER_LIQUIDITY = Histogram(
    "solver_liquidity_sources_per_auction",
    "Number of liquidity sources per auction",
    buckets=[1, 5, 10, 25, 50, 100, 250, 500, 1000],
)

SOLVER_ERRORS = Counter(
    "solver_errors_total", "Total number of solver errors", ["error_type"]
)

SOLVER_PERFORMANCE = Histogram(
    "solver_performance_seconds", "Solver performance metrics", ["operation"]
)


class MetricsCollector:
    """Metrics collection helper."""

    def __init__(self):
        self.start_time = time.time()

    def record_request(self, method: str, endpoint: str, status: str, duration: float):
        """Record a request metric."""
        REQUEST_COUNT.labels(method=method, endpoint=endpoint, status=status).inc()
        REQUEST_DURATION.labels(method=method, endpoint=endpoint).observe(duration)

    def record_solution(self, status: str):
        """Record a solution generation metric."""
        SOLVER_SOLUTIONS.labels(status=status).inc()

    def record_auction_orders(self, order_count: int):
        """Record the number of orders in an auction."""
        SOLVER_ORDERS.observe(order_count)

    def record_auction_tokens(self, token_count: int):
        """Record the number of tokens in an auction."""
        SOLVER_TOKENS.observe(token_count)

    def record_auction_liquidity(self, liquidity_count: int):
        """Record the number of liquidity sources in an auction."""
        SOLVER_LIQUIDITY.observe(liquidity_count)

    def record_error(self, error_type: str):
        """Record an error metric."""
        SOLVER_ERRORS.labels(error_type=error_type).inc()

    def record_performance(self, operation: str, duration: float):
        """Record a performance metric."""
        SOLVER_PERFORMANCE.labels(operation=operation).observe(duration)

    def set_active_requests(self, count: int):
        """Set the number of active requests."""
        ACTIVE_REQUESTS.set(count)

    def get_metrics(self) -> str:
        """Get Prometheus metrics in text format."""
        return generate_latest().decode("utf-8")

    def get_metrics_content_type(self) -> str:
        """Get the content type for metrics."""
        return CONTENT_TYPE_LATEST


# Global metrics collector
_metrics = MetricsCollector()


def get_metrics() -> str:
    """Get Prometheus metrics."""
    return _metrics.get_metrics()


def get_metrics_content_type() -> str:
    """Get metrics content type."""
    return _metrics.get_metrics_content_type()


def record_request(method: str, endpoint: str, status: str, duration: float):
    """Record a request metric."""
    _metrics.record_request(method, endpoint, status, duration)


def record_solution(status: str):
    """Record a solution generation metric."""
    _metrics.record_solution(status)


def record_auction_orders(order_count: int):
    """Record auction order count."""
    _metrics.record_auction_orders(order_count)


def record_auction_tokens(token_count: int):
    """Record auction token count."""
    _metrics.record_auction_tokens(token_count)


def record_auction_liquidity(liquidity_count: int):
    """Record auction liquidity count."""
    _metrics.record_auction_liquidity(liquidity_count)


def record_error(error_type: str):
    """Record an error metric."""
    _metrics.record_error(error_type)


def record_performance(operation: str, duration: float):
    """Record a performance metric."""
    _metrics.record_performance(operation, duration)


def set_active_requests(count: int):
    """Set active request count."""
    _metrics.set_active_requests(count)

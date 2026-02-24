"""
Tests for API health and metrics endpoints.

This module contains tests for the health check and metrics endpoints.
"""

import pytest
from fastapi.testclient import TestClient
from src.api.app import app


client = TestClient(app)


class TestHealthEndpoint:
    """Test health check endpoint."""

    def test_health_endpoint(self):
        """Test health check endpoint returns 200."""
        response = client.get("/healthz")
        assert response.status_code == 200

        data = response.json()
        assert "status" in data
        assert "timestamp" in data
        assert "version" in data
        assert "service" in data

    def test_health_endpoint_structure(self):
        """Test health check endpoint response structure."""
        response = client.get("/healthz")
        data = response.json()

        # Check required fields
        assert "status" in data
        assert "timestamp" in data
        assert "version" in data
        assert "service" in data

        # Check field types
        assert isinstance(data["status"], str)
        assert isinstance(data["timestamp"], str)
        assert isinstance(data["version"], str)
        assert isinstance(data["service"], str)

    def test_health_endpoint_values(self):
        """Test health check endpoint field values."""
        response = client.get("/healthz")
        data = response.json()

        assert data["status"] == "healthy"
        assert data["version"] == "1.0.0"
        assert data["service"] == "cow-solver-baseline"


class TestMetricsEndpoint:
    """Test metrics endpoint."""

    def test_metrics_endpoint(self):
        """Test metrics endpoint returns 200."""
        response = client.get("/metrics")
        assert response.status_code == 200

    def test_metrics_endpoint_content_type(self):
        """Test metrics endpoint content type."""
        response = client.get("/metrics")
        assert (
            response.headers["content-type"]
            == "text/plain; version=1.0.0; charset=utf-8"
        )

    def test_metrics_endpoint_content(self):
        """Test metrics endpoint content."""
        response = client.get("/metrics")
        content = response.text

        # Check for Prometheus metrics format
        assert "# HELP" in content
        assert "# TYPE" in content
        assert "solver_requests_total" in content
        assert "solver_request_duration_seconds" in content
        assert "solver_active_requests" in content
        assert "solver_solutions_total" in content
        assert "solver_orders_per_auction" in content


class TestRootEndpoint:
    """Test root endpoint."""

    def test_root_endpoint(self):
        """Test root endpoint returns 200."""
        response = client.get("/")
        assert response.status_code == 200

        data = response.json()
        assert "service" in data
        assert "version" in data
        assert "status" in data

    def test_root_endpoint_structure(self):
        """Test root endpoint response structure."""
        response = client.get("/")
        data = response.json()

        # Check required fields
        assert "service" in data
        assert "version" in data
        assert "status" in data

        # Check field types
        assert isinstance(data["service"], str)
        assert isinstance(data["version"], str)
        assert isinstance(data["status"], str)

    def test_root_endpoint_values(self):
        """Test root endpoint field values."""
        response = client.get("/")
        data = response.json()

        assert data["service"] == "CoW Protocol Solver"
        assert data["version"] == "1.0.0"
        assert data["status"] == "running"


class TestCORSHeaders:
    """Test CORS headers."""

    def test_cors_headers(self):
        """Test CORS headers are present."""
        response = client.options("/")
        assert response.status_code == 200

        # Check CORS headers
        assert "access-control-allow-origin" in response.headers
        assert "access-control-allow-methods" in response.headers
        assert "access-control-allow-headers" in response.headers


class TestErrorHandling:
    """Test error handling."""

    def test_404_endpoint(self):
        """Test 404 for non-existent endpoint."""
        response = client.get("/nonexistent")
        assert response.status_code == 404

    def test_405_method_not_allowed(self):
        """Test 405 for unsupported method."""
        response = client.put("/healthz")
        assert response.status_code == 405

    def test_422_validation_error(self):
        """Test 422 for validation error."""
        response = client.post("/solve", json={"invalid": "data"})
        assert response.status_code == 422

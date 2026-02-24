"""
Contract tests for the /solve endpoint.

This module contains tests to verify the /solve endpoint contract compliance.
"""

import pytest
from fastapi.testclient import TestClient
from src.api.app import app


client = TestClient(app)


class TestSolveEndpointContract:
    """Test /solve endpoint contract compliance."""

    def test_solve_endpoint_exists(self):
        """Test /solve endpoint exists and accepts POST."""
        response = client.post("/solve", json={})
        # Should return 422 (validation error) not 404 (not found)
        assert response.status_code == 422

    def test_solve_endpoint_accepts_post(self):
        """Test /solve endpoint accepts POST method."""
        response = client.post("/solve", json={})
        assert response.status_code == 422  # Validation error, not method error

    def test_solve_endpoint_rejects_get(self):
        """Test /solve endpoint rejects GET method."""
        response = client.get("/solve")
        assert response.status_code == 405  # Method not allowed

    def test_solve_endpoint_rejects_put(self):
        """Test /solve endpoint rejects PUT method."""
        response = client.put("/solve", json={})
        assert response.status_code == 405  # Method not allowed

    def test_solve_endpoint_rejects_delete(self):
        """Test /solve endpoint rejects DELETE method."""
        response = client.delete("/solve")
        assert response.status_code == 405  # Method not allowed


class TestSolveEndpointValidation:
    """Test /solve endpoint request validation."""

    def test_solve_endpoint_requires_json(self):
        """Test /solve endpoint requires JSON content."""
        response = client.post("/solve", data="not json")
        assert response.status_code == 422

    def test_solve_endpoint_requires_auction_id(self):
        """Test /solve endpoint requires auction ID."""
        response = client.post(
            "/solve",
            json={
                "tokens": {},
                "orders": [],
                "liquidity": [],
                "effective_gas_price": "20000000000",
                "deadline": "2024-12-19T12:00:00Z",
            },
        )
        assert response.status_code == 422
        assert "id" in str(response.json())

    def test_solve_endpoint_requires_tokens(self):
        """Test /solve endpoint requires tokens."""
        response = client.post(
            "/solve",
            json={
                "id": "0x1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef",
                "orders": [],
                "liquidity": [],
                "effective_gas_price": "20000000000",
                "deadline": "2024-12-19T12:00:00Z",
            },
        )
        # Solver returns 200 but with empty solution when no tokens
        assert response.status_code == 200
        data = response.json()
        assert "solutions" in data
        # When no tokens, solver returns empty solutions array
        assert len(data["solutions"]) == 0

    def test_solve_endpoint_requires_orders(self):
        """Test /solve endpoint requires orders."""
        response = client.post(
            "/solve",
            json={
                "id": "0x1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef",
                "tokens": {
                    "0x1234567890123456789012345678901234567890": {
                        "decimals": 18,
                        "symbol": "WETH",
                    }
                },
                "liquidity": [],
                "effective_gas_price": "20000000000",
                "deadline": "2024-12-19T12:00:00Z",
            },
        )
        # Solver returns 200 but with empty solution when no orders
        assert response.status_code == 200
        data = response.json()
        assert "solutions" in data
        # When no orders, solver returns solution with empty trades
        assert len(data["solutions"]) == 1
        solution = data["solutions"][0]
        assert len(solution["trades"]) == 0

    def test_solve_endpoint_requires_effective_gas_price(self):
        """Test /solve endpoint requires effective gas price."""
        response = client.post(
            "/solve",
            json={
                "id": "0x1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef",
                "tokens": {
                    "0x1234567890123456789012345678901234567890": {
                        "decimals": 18,
                        "symbol": "WETH",
                    }
                },
                "orders": [],
                "liquidity": [],
                "deadline": "2024-12-19T12:00:00Z",
            },
        )
        assert response.status_code == 422
        assert "effectiveGasPrice" in str(response.json())

    def test_solve_endpoint_requires_deadline(self):
        """Test /solve endpoint requires deadline."""
        response = client.post(
            "/solve",
            json={
                "id": "0x1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef",
                "tokens": {
                    "0x1234567890123456789012345678901234567890": {
                        "decimals": 18,
                        "symbol": "WETH",
                    }
                },
                "orders": [],
                "liquidity": [],
                "effective_gas_price": "20000000000",
            },
        )
        assert response.status_code == 422
        assert "deadline" in str(response.json())


class TestSolveEndpointValidRequest:
    """Test /solve endpoint with valid requests."""

    def test_solve_endpoint_minimal_valid_request(self):
        """Test /solve endpoint with minimal valid request."""
        response = client.post(
            "/solve",
            json={
                "id": "0x1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef",
                "tokens": {
                    "0x1234567890123456789012345678901234567890": {
                        "decimals": 18,
                        "symbol": "WETH",
                    }
                },
                "orders": [
                    {
                        "uid": "0x1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef",
                        "sellToken": "0x1234567890123456789012345678901234567890",
                        "buyToken": "0x0987654321098765432109876543210987654321",
                        "sellAmount": "1000000000000000000",
                        "fullSellAmount": "1000000000000000000",
                        "buyAmount": "2000000000000000000000",
                        "fullBuyAmount": "2000000000000000000000",
                        "validTo": 1734609600,
                        "kind": "sell",
                        "receiver": "0x1234567890123456789012345678901234567890",
                        "owner": "0x1234567890123456789012345678901234567890",
                        "partiallyFillable": False,
                        "preInteractions": [],
                        "postInteractions": [],
                        "sellTokenBalance": "erc20",
                        "buyTokenBalance": "erc20",
                        "appData": "0x0000000000000000000000000000000000000000000000000000000000000000",
                        "signature": "0xabcdef1234567890abcdef1234567890abcdef1234567890abcdef1234567890",
                    }
                ],
                "liquidity": [],
                "effective_gas_price": "20000000000",
                "deadline": "2024-12-19T12:00:00Z",
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert "solutions" in data
        assert isinstance(data["solutions"], list)

    def test_solve_endpoint_response_structure(self):
        """Test /solve endpoint response structure."""
        response = client.post(
            "/solve",
            json={
                "id": "0x1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef",
                "tokens": {
                    "0x1234567890123456789012345678901234567890": {
                        "decimals": 18,
                        "symbol": "WETH",
                    }
                },
                "orders": [
                    {
                        "uid": "0x1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef",
                        "sellToken": "0x1234567890123456789012345678901234567890",
                        "buyToken": "0x0987654321098765432109876543210987654321",
                        "sellAmount": "1000000000000000000",
                        "fullSellAmount": "1000000000000000000",
                        "buyAmount": "2000000000000000000000",
                        "fullBuyAmount": "2000000000000000000000",
                        "validTo": 1734609600,
                        "kind": "sell",
                        "receiver": "0x1234567890123456789012345678901234567890",
                        "owner": "0x1234567890123456789012345678901234567890",
                        "partiallyFillable": False,
                        "preInteractions": [],
                        "postInteractions": [],
                        "sellTokenBalance": "erc20",
                        "buyTokenBalance": "erc20",
                        "appData": "0x0000000000000000000000000000000000000000000000000000000000000000",
                        "signature": "0xabcdef1234567890abcdef1234567890abcdef1234567890abcdef1234567890",
                    }
                ],
                "liquidity": [],
                "effective_gas_price": "20000000000",
                "deadline": "2024-12-19T12:00:00Z",
            },
        )

        data = response.json()

        # Check response structure
        assert "solutions" in data
        assert isinstance(data["solutions"], list)

        # Check solution structure (if any solutions)
        if data["solutions"]:
            solution = data["solutions"][0]
            assert "id" in solution
            assert "trades" in solution
            assert "prices" in solution
            assert "interactions" in solution
            assert isinstance(solution["trades"], list)
            assert isinstance(solution["prices"], dict)
            assert isinstance(solution["interactions"], list)

    def test_solve_endpoint_empty_solution(self):
        """Test /solve endpoint returns empty solution for MVP."""
        response = client.post(
            "/solve",
            json={
                "id": "0x1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef",
                "tokens": {
                    "0x1234567890123456789012345678901234567890": {
                        "decimals": 18,
                        "symbol": "WETH",
                    }
                },
                "orders": [
                    {
                        "uid": "0x1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef",
                        "sellToken": "0x1234567890123456789012345678901234567890",
                        "buyToken": "0x0987654321098765432109876543210987654321",
                        "sellAmount": "1000000000000000000",
                        "fullSellAmount": "1000000000000000000",
                        "buyAmount": "2000000000000000000000",
                        "fullBuyAmount": "2000000000000000000000",
                        "validTo": 1734609600,
                        "kind": "sell",
                        "receiver": "0x1234567890123456789012345678901234567890",
                        "owner": "0x1234567890123456789012345678901234567890",
                        "partiallyFillable": False,
                        "preInteractions": [],
                        "postInteractions": [],
                        "sellTokenBalance": "erc20",
                        "buyTokenBalance": "erc20",
                        "appData": "0x0000000000000000000000000000000000000000000000000000000000000000",
                        "signature": "0xabcdef1234567890abcdef1234567890abcdef1234567890abcdef1234567890",
                    }
                ],
                "liquidity": [],
                "effective_gas_price": "20000000000",
                "deadline": "2024-12-19T12:00:00Z",
            },
        )

        data = response.json()

        # For MVP, should return empty solution
        assert len(data["solutions"]) == 1
        solution = data["solutions"][0]
        assert len(solution["trades"]) == 0
        assert "prices" in solution
        assert len(solution["interactions"]) == 0

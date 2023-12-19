"""Tests for the API endpoints."""

from fastapi.testclient import TestClient
from src._server import app

client = TestClient(app)


def test_notify() -> None:
    """Check notify endpoint."""
    response = client.post("/notify", json={"note": "everything is fine"})
    assert response.status_code == 200
    assert response.json() is True


def test_solve() -> None:
    """Check solve endpoints."""
    auction = {
        "id": "string",
        "tokens": {
            "additionalProp1": {
                "decimals": 0,
                "symbol": "string",
                "referencePrice": "1234567890",
                "availableBalance": "1234567890",
                "trusted": True,
            },
            "additionalProp2": {
                "decimals": 0,
                "symbol": "string",
                "referencePrice": "1234567890",
                "availableBalance": "1234567890",
                "trusted": True,
            },
            "additionalProp3": {
                "decimals": 0,
                "symbol": "string",
                "referencePrice": "1234567890",
                "availableBalance": "1234567890",
                "trusted": True,
            },
        },
        "orders": [
            {
                "uid": "0x30cff40d9f60caa68a37f0ee73253ad6ad72b45580c945fe3ab67596476937197854163b1b0d24e77dca702b97b5cc33e0f83dcb626122a6",  # pylint: disable=line-too-long
                "sellToken": "0xDEf1CA1fb7FBcDC777520aa7f396b4E015F497aB",
                "buyToken": "0xDEf1CA1fb7FBcDC777520aa7f396b4E015F497aB",
                "sellAmount": "1234567890",
                "buyAmount": "1234567890",
                "feeAmount": "1234567890",
                "kind": "sell",
                "partiallyFillable": True,
                "class": "market",
            }
        ],
        "liquidity": [
            {
                "kind": "constantproduct",
                "tokens": {
                    "additionalProp1": {"balance": "1234567890"},
                    "additionalProp2": {"balance": "1234567890"},
                    "additionalProp3": {"balance": "1234567890"},
                },
                "fee": "13.37",
                "id": "string",
                "address": "0x0000000000000000000000000000000000000000",
                "gasEstimate": "1234567890",
            },
            {
                "kind": "weightedproduct",
                "tokens": {
                    "additionalProp1": {
                        "balance": "1234567890",
                        "scalingFactor": "1234567890",
                        "weight": "13.37",
                    },
                    "additionalProp2": {
                        "balance": "1234567890",
                        "scalingFactor": "1234567890",
                        "weight": "13.37",
                    },
                    "additionalProp3": {
                        "balance": "1234567890",
                        "scalingFactor": "1234567890",
                        "weight": "13.37",
                    },
                },
                "fee": "13.37",
                "id": "string",
                "address": "0x0000000000000000000000000000000000000000",
                "gasEstimate": "1234567890",
            },
            {
                "kind": "stable",
                "tokens": {
                    "additionalProp1": {
                        "balance": "1234567890",
                        "scalingFactor": "1234567890",
                    },
                    "additionalProp2": {
                        "balance": "1234567890",
                        "scalingFactor": "1234567890",
                    },
                    "additionalProp3": {
                        "balance": "1234567890",
                        "scalingFactor": "1234567890",
                    },
                },
                "amplificationParameter": "13.37",
                "fee": "13.37",
                "id": "string",
                "address": "0x0000000000000000000000000000000000000000",
                "gasEstimate": "1234567890",
            },
            {
                "kind": "concentratedliquidity",
                "tokens": ["0xDEf1CA1fb7FBcDC777520aa7f396b4E015F497aB"],
                "sqrtPrice": "1234567890",
                "liquidity": "1234567890",
                "tick": 0,
                "liquidityNet": {
                    "additionalProp1": "1234567890",
                    "additionalProp2": "1234567890",
                    "additionalProp3": "1234567890",
                },
                "fee": "13.37",
                "id": "string",
                "address": "0x0000000000000000000000000000000000000000",
                "gasEstimate": "1234567890",
            },
            {
                "kind": "limitorder",
                "hash": "0x1e66721bb1bd77d2641c77ea1d61e8abb92bf69c64fcc90c2c6ad518d1b50db1",
                "makerToken": "0xDEf1CA1fb7FBcDC777520aa7f396b4E015F497aB",
                "takerToken": "0xDEf1CA1fb7FBcDC777520aa7f396b4E015F497aB",
                "makerAmount": "1234567890",
                "takerAmount": "1234567890",
                "takerTokenFeeAmount": "1234567890",
                "id": "string",
                "address": "0x0000000000000000000000000000000000000000",
                "gasEstimate": "1234567890",
            },
        ],
        "effectiveGasPrice": "1234567890",
        "deadline": "1970-01-01T00:00:00.000Z",
    }
    trivial_solution = {
        "id": "123",
        "trades": [],
        "prices": {},
        "interactions": [],
        "solver": "solvertemplate",
        "score": "0",
        "weth": "0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2",
    }

    response = client.post("/solve", json=auction)
    assert response.json() == trivial_solution

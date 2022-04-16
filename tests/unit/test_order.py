import unittest

from src.models.order import Order, OrderMatchType
from src.models.token import Token


class MyTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.token_a = "0x6a023ccd1ff6f2045c3309768ead9e68f978f6e1"
        self.token_b = "0x177127622c4a00f3d409b75571e12cb3c8973d3c"
        self.token_c = "0x1111111111111111111111111111111111111111"

    def overlapping_orders(self):
        order1 = Order.from_dict(
            "1",
            {
                "sell_token": self.token_a,
                "buy_token": self.token_b,
                "sell_amount": "12",
                "buy_amount": "100",
                "allow_partial_fill": False,
                "is_sell_order": True,
                "fee": {
                    "amount": "115995469750",
                    "token": "0x6a023ccd1ff6f2045c3309768ead9e68f978f6e1",
                },
                "cost": {
                    "amount": "321627750000000",
                    "token": "0xe91d153e0b41518a2ce8dd3d7944fa863463a97d",
                },
                "is_liquidity_order": False,
                "mandatory": False,
                "has_atomic_execution": False,
            },
        )
        order2 = Order.from_dict(
            "2",
            {
                "sell_token": self.token_b,
                "buy_token": self.token_a,
                "sell_amount": "100",
                "buy_amount": "10",
                "allow_partial_fill": False,
                "is_sell_order": True,
                "fee": {
                    "amount": "115995469750",
                    "token": "0x6a023ccd1ff6f2045c3309768ead9e68f978f6e1",
                },
                "cost": {
                    "amount": "321627750000000",
                    "token": "0xe91d153e0b41518a2ce8dd3d7944fa863463a97d",
                },
                "is_liquidity_order": False,
                "mandatory": False,
                "has_atomic_execution": False,
            },
        )

        return order1, order2

    def test_overlaps(self):

        order1, order2 = self.overlapping_orders()
        self.assertTrue(order1.overlaps(order2))

        # Change the buy amount to make it so the orders don't overlap.
        old_buy_amount = order1.buy_amount

        order1.buy_amount *= 10
        self.assertFalse(order1.overlaps(order2))

        # Set the buy amount back and change the token.
        order1.buy_amount = old_buy_amount
        self.assertTrue(order1.overlaps(order2))
        token_c = "0x1111111111111111111111111111111111111111"
        order1.buy_token = Token(token_c)
        self.assertFalse(order1.overlaps(order2))

    def test_match_type(self):
        order1, order2 = self.overlapping_orders()

        self.assertEqual(order1.match_type(order2), OrderMatchType.BOTH_FILLED)

        # Make order1 half the size
        order1.buy_amount /= 2
        order1.sell_amount /= 2

        self.assertEqual(order1.match_type(order2), OrderMatchType.LHS_FILLED)
        # Reverse the orders to get RHS Filled
        self.assertEqual(order2.match_type(order1), OrderMatchType.RHS_FILLED)

        order1.buy_token = Token(self.token_c)

        self.assertIsNone(order1.match_type(order2))


if __name__ == "__main__":
    unittest.main()

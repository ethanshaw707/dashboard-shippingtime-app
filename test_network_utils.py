import unittest

import pandas as pd

from network_utils import compute_best_origin_map


class TestComputeBestOriginMap(unittest.TestCase):
    def test_returns_best_origin_per_destination(self):
        # Arrange
        df = pd.DataFrame(
            [
                {"Destination": "A, AA", "FromAddress": "Origin1", "ShippingTimeDays": 1},
                {"Destination": "A, AA", "FromAddress": "Origin2", "ShippingTimeDays": 2},
                {"Destination": "B, BB", "FromAddress": "Origin1", "ShippingTimeDays": 2},
                {"Destination": "B, BB", "FromAddress": "Origin2", "ShippingTimeDays": 2},
            ]
        )

        # Act
        result = compute_best_origin_map(df)

        # Assert
        self.assertEqual(result["A, AA"], "Origin1")
        self.assertEqual(result["B, BB"], "Origin1, Origin2")


if __name__ == "__main__":
    unittest.main()

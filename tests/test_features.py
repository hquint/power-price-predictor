import pandas as pd
from app.features.engineer import create_features
import unittest


class TestFeatureEngineering(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame(
            {
                "hour_of_day": [0, 12],
                "wind_speed": [4.0, 5.0],
                "temperature": [2.0, 8.0],
            }
        )

    def test_columns_added(self):
        """
        Test that create_features adds the expected columns to the DataFrame.
        """
        result = create_features(self.df)
        self.assertIn("hour_sin", result.columns)
        self.assertIn("hour_cos", result.columns)
        self.assertIn("wind_temp_ratio", result.columns)

    def test_no_missing_values(self):
        result = create_features(self.df)
        self.assertFalse(result.isnull().any().any())


if __name__ == "__main__":
    unittest.main()

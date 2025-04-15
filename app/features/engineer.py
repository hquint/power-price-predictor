import pandas as pd
import numpy as np


class FeatureEngineering:
    def __init__(self, lags: list[int] = None, rolling: list[int] = None):
        """
        Initialize the FeatureEngineering class.

        Parameters:
        - lags: List of lag values for creating lag features
        - rolling: List of rolling window sizes for creating rolling features
        """
        self.lags = lags or [1, 2, 3]
        self.rolling = rolling or [3, 6]

    def add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:

        df["hour_sin"] = np.sin(2 * np.pi * df["hour_of_day"] / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df["hour_of_day"] / 24)
        df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

        return df

    def add_lag_features(self, df: pd.DataFrame, target_column: str) -> pd.DataFrame:
        for lag in self.lags:
            df[f"{target_column}_lag{lag}"] = df[target_column].shift(lag)

        return df

    def add_rolling_features(
        self, df: pd.DataFrame, target_column: str
    ) -> pd.DataFrame:
        for window in self.rolling:
            df[f"{target_column}_rolling_mean_{window}"] = (
                df[target_column].rolling(window=window).mean()
            )
            df[f"{target_column}_rolling_std_{window}"] = (
                df[target_column].rolling(window=window).std()
            )

        return df

    def add_cross_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df["wind_temp_ratio"] = df["wind_speed"] / (df["temperature"] + 1e-6)

        return df

    def transform(
        self, df: pd.DataFrame, target_column: str = "price_next_hour"
    ) -> pd.DataFrame:
        """
        Apply feature engineering transformations to the DataFrame.

        Parameters:
        - df: Input DataFrame
        - target_column: Target column for lag and rolling features

        Returns:
        - Transformed DataFrame with new features
        """
        df = df.copy()
        df = self.add_time_features(df)
        df = self.add_lag_features(df, target_column)
        df = self.add_rolling_features(df, target_column)
        df = self.add_cross_features(df)
        df.dropna(inplace=True)

        return df

    def get_feature_names(
        self, df: pd.DataFrame, target_column: str = "price_next_hour"
    ) -> list[str]:
        """
        Get the names of the features created by the feature engineering process.

        Parameters:
        - df: Input DataFrame
        - target_column: Target column for lag and rolling features

        Returns:
        - List of feature names
        """
        return [col for col in df.columns if col != target_column]

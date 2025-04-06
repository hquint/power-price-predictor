import pandas as pd
import numpy as np


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    # Create time-based features
    df = df.copy()

    df["hour_sin"] = np.sin(2 * np.pi * df["hour_of_day"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour_of_day"] / 24)

    df["wind_temp_ratio"] = df["wind_speed"] / (df["temperature"] + 1)

    return df

import pandas as pd
import numpy as np
import pytest
from app.features.engineer import FeatureEngineering


@pytest.fixture
def sample_df():
    # Minimal example that can survive lag and rolling
    df = pd.DataFrame(
        {
            "hour_of_day": list(range(10)),
            "day_of_week": [0, 1, 2, 3, 4, 5, 6, 0, 1, 2],
            "temperature": np.random.normal(15, 5, 10),
            "wind_speed": np.random.normal(5, 2, 10),
            "demand_forecast": np.random.normal(3000, 300, 10),
            "price_next_hour": np.random.normal(50, 10, 10),
        }
    )
    return df


def test_transform_adds_expected_columns(sample_df):
    fe = FeatureEngineering(lags=[1], rolling_windows=[2])
    df_out = fe.transform(sample_df)

    expected_cols = [
        "hour_of_day",
        "day_of_week",
        "temperature",
        "wind_speed",
        "demand_forecast",
        "price_next_hour",
        "hour_sin",
        "hour_cos",
        "is_weekend",
        "price_next_hour_lag1",
        "price_next_hour_rolling_mean_2",
        "price_next_hour_rolling_std_2",
        "wind_temp_ratio",
    ]
    for col in expected_cols:
        assert col in df_out.columns


def test_no_nans_after_transform(sample_df):
    fe = FeatureEngineering(lags=[1], rolling_windows=[2])
    df_out = fe.transform(sample_df)
    assert not df_out.isnull().any().any()


def test_get_feature_names(sample_df):
    fe = FeatureEngineering(lags=[1], rolling_windows=[2])
    df_out = fe.transform(sample_df)
    feature_names = fe.get_feature_names(df_out)
    assert "price_next_hour" not in feature_names
    assert isinstance(feature_names, list)
    assert all(isinstance(name, str) for name in feature_names)


def test_row_count_after_transform(sample_df):
    lags = [1]
    rolling_windows = [2]
    fe = FeatureEngineering(lags=lags, rolling_windows=rolling_windows)
    df_out = fe.transform(sample_df)
    max_lag = max(lags)
    max_rolling = max(rolling_windows)
    expected_rows = len(sample_df) - max(max_lag, max_rolling - 1)
    assert len(df_out) == expected_rows

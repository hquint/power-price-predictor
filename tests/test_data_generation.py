import pandas as pd
import pytest
from data.generate_real_data import merge_and_save


@pytest.fixture
def mock_data():
    timestamps = pd.date_range("2024-01-01", periods=3, freq="h", tz="Europe/Berlin")
    df_entsoe = pd.DataFrame(
        {
            "timestamp": timestamps,
            "price_day_ahead": [50.1, 52.3, 48.9],
            "actual_load": [60000, 59000, 61000],
        }
    )
    df_weather = pd.DataFrame(
        {
            "timestamp": timestamps,
            "temperature_2m": [2.0, 3.0, 2.5],
            "wind_speed_10m": [5.0, 4.5, 1.2],
            "apparent_temperature": [1.0, 2.5, 3],
            "is_day": [1, 1, 0],
        }
    )
    return df_entsoe, df_weather


def test_merge_and_save(mocker, mock_data, tmp_path):
    df_entsoe, df_weather = mock_data

    # Patch the fetch functions using mocker fixture
    mocker.patch("data.generate_real_data.fetch_entsoe_data", return_value=df_entsoe)
    mocker.patch(
        "data.generate_real_data.fetch_open_meteo_data", return_value=df_weather
    )

    output_path = tmp_path / "test_output.csv"
    merge_and_save(output_path=output_path)

    # Check output
    df = pd.read_csv(output_path)

    assert not df.empty
    assert set(df.columns) == {
        "timestamp",
        "price_day_ahead",
        "actual_load",
        "temperature_2m",
        "wind_speed_10m",
        "apparent_temperature",
        "is_day",
    }
    assert len(df) == 3
    assert len(df.columns) == 7

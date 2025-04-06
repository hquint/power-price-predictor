# data/generate_realistic_mock_data.py
import pandas as pd
import numpy as np


def generate_realistic_mock_data(days=90):
    np.random.seed(42)

    # 90 days of hourly data
    period = pd.date_range(start="2024-01-01", periods=24 * days, freq="h")
    df = pd.DataFrame(index=period)
    df["hour_of_day"] = df.index.hour
    df["day_of_week"] = df.index.dayofweek

    # Add sinusoidal patterns for seasonality + noise
    df["temperature"] = (
        10
        + 10 * np.sin(2 * np.pi * df.index.dayofyear / 365)
        + np.random.normal(0, 2, len(df))
    )
    df["wind_speed"] = (
        5
        + 2 * np.sin(2 * np.pi * df.index.hour / 24)
        + np.random.normal(0, 0.5, len(df))
    )
    df["demand_forecast"] = (
        3000
        + 300 * np.cos(2 * np.pi * df.index.hour / 24)
        + np.random.normal(0, 100, len(df))
    )

    # Simulate power price with daily + weekly patterns
    base_price = 50
    daily_cycle = 10 * np.sin(2 * np.pi * df.index.hour / 24)
    weekly_cycle = 5 * np.cos(2 * np.pi * df.index.dayofweek / 7)
    noise = np.random.normal(0, 3, len(df))
    df["price_next_hour"] = base_price + daily_cycle + weekly_cycle + noise

    df.reset_index(inplace=True)
    df.rename(columns={"index": "timestamp"}, inplace=True)

    df.to_csv("data/power_market_data.csv", index=False)
    print("✅ Realistic mock data generated!")


if __name__ == "__main__":
    generate_realistic_mock_data()

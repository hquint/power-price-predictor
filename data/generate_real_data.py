import os
from dotenv import load_dotenv
from entsoe import EntsoePandasClient
import requests
import pandas as pd

# Config
country_code = "DE_LU"
latitude = 52.52  # Berlin
longitude = 13.405
start_date = "2024-04-01"
end_date = "2024-04-15"
tz = "Europe/Berlin"

# Convert to Timestamps
start = pd.Timestamp(start_date, tz=tz)
end = pd.Timestamp(end_date, tz=tz)


def fetch_entsoe_data():
    # Load ENTSO-E API key from .env
    load_dotenv()
    api_key = os.getenv("ENTSOE_API_KEY")
    client = EntsoePandasClient(api_key=api_key)

    print("📡 Fetching day-ahead prices...")

    prices = client.query_day_ahead_prices(country_code, start=start, end=end)

    print("📡 Fetching actual total load...")
    load = client.query_load(country_code, start=start, end=end)

    df_entsoe = pd.DataFrame(
        {
            "timestamp": prices.index,
            "price_day_ahead": prices.values,
            "actual_load": load.reindex(prices.index)[
                "Actual Load"
            ].values,  # align by timestamp
        }
    )
    return df_entsoe


def fetch_open_meteo_data():
    print("🌤 Fetching historical weather from Open-Meteo...")
    url = "https://archive-api.open-meteo.com/v1/era5"

    hourly_params = [
        "temperature_2m",
        "wind_speed_10m",
        "apparent_temperature",
        "is_day",
    ]

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": hourly_params,
        "timezone": tz,
    }

    response = requests.get(url, params=params)
    data = response.json()

    # df_weather = pd.DataFrame({
    #     "timestamp": pd.to_datetime(data["hourly"]["time"]),
    #     "temperature_2m": data["hourly"]["temperature_2m"],
    #     "wind_speed_10m": data["hourly"]["wind_speed_10m"]
    # })

    df_weather = pd.DataFrame(data["hourly"])

    df_weather["timestamp"] = pd.to_datetime(df_weather["time"]).dt.tz_localize(
        "Europe/Berlin"
    )
    df_weather.set_index("timestamp", inplace=True)
    df_weather.drop(columns="time", inplace=True)

    return df_weather


def merge_and_save(output_path="data/power_market_real_data.csv"):
    df_prices = fetch_entsoe_data()
    df_weather = fetch_open_meteo_data()

    print("🔄 Merging datasets...")
    df = pd.merge(df_prices, df_weather, on="timestamp", how="inner")
    df.to_csv(output_path, index=False)
    print("✅ Saved to data/power_market_real_data.csv")
    print(df.head())


if __name__ == "__main__":
    merge_and_save()

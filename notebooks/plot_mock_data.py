# %%
import pandas as pd
import matplotlib.pyplot as plt

# %%
df = pd.read_csv("../data/power_market_data.csv", parse_dates=["timestamp"])
df.set_index("timestamp", inplace=True)

fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
df["temperature"].plot(ax=axes[0], title="Temperature (°C)")
df["wind_speed"].plot(ax=axes[1], title="Wind Speed (m/s)")
df["demand_forecast"].plot(ax=axes[2], title="Demand Forecast (MW)")
df["price_next_hour"].plot(ax=axes[3], title="Power Price (€/MWh)")

plt.tight_layout()
plt.show()

from pathlib import Path

DATA_PATH = Path("data/power_market_data.csv")


def get_model_path(model_type: str) -> Path:
    return Path(f"app/models/{model_type}_pipeline.pkl")

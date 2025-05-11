import pytest
import pandas as pd
import numpy as np
from app.models.pipeline import build_pipeline


@pytest.fixture
def sample_data():
    X = pd.DataFrame(
        {
            "temperature": np.random.normal(15, 5, 10),
            "wind_speed": np.random.normal(5, 2, 10),
            "demand_forecast": np.random.normal(3000, 500, 10),
            "is_weekend": [0, 1] * 5,
        }
    )
    y = np.random.normal(50, 5, 10)
    return X, y


def test_build_pipeline_rf(sample_data):
    X, y = sample_data
    pipeline = build_pipeline(
        numeric_features=["temperature", "wind_speed", "demand_forecast"],
        binary_features=["is_weekend"],
        model_type="rf",  # random forest
    )
    pipeline.fit(X, y)
    preds = pipeline.predict(X)
    assert len(preds) == len(X)


def test_build_pipeline_gb(sample_data):
    X, y = sample_data
    pipeline = build_pipeline(
        numeric_features=["temperature", "wind_speed", "demand_forecast"],
        binary_features=["is_weekend"],
        model_type="gb",  # gradient boosting
    )
    pipeline.fit(X, y)
    preds = pipeline.predict(X)
    assert len(preds) == len(X)


def test_build_pipeline_invalid_model():
    with pytest.raises(ValueError, match="Unsupported model type"):
        build_pipeline(numeric_features=["x"], binary_features=["y"], model_type="xyz")

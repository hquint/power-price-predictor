import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

from app.config import DATA_PATH, MODEL_PATH
from app.features.engineer import create_features
from app.models.pipeline import build_pipeline


def train():
    """
    1. Load the dataset from a CSV file.
    2. Create features using the create_features function.
    3. Split the dataset into training and testing sets.
    4. Build a machine learning pipeline using the build_pipeline function.
    5. Train the pipeline on the training set.
    6. Evaluate the model on the test set using MAE and R² metrics.
    7. Save the trained model to a file.
    """

    # Load the dataset
    df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])
    df = create_features(df)

    features = [
        "temperature",
        "wind_speed",
        "demand_forecast",
        "hour_sin",
        "hour_cos",
        "wind_temp_ratio",
    ]
    target = "price_next_hour"

    # Ensure the target variable is not in the features
    X = df[features]
    y = df[target]

    # Split the data into training and testing sets
    # Using shuffle=False to maintain the time series order
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    # Build the pipeline and train the model
    pipeline = build_pipeline(features)
    pipeline.fit(X_train, y_train)

    # Evaluate the model
    y_pred = pipeline.predict(X_test)
    print(f"MAE: {mean_absolute_error(y_test, y_pred):.2f}")
    print(f"R² : {r2_score(y_test, y_pred):.2f}")

    joblib.dump(pipeline, MODEL_PATH)
    print(f"✅ Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    train()

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

from app.config import DATA_PATH, get_model_path
from app.features.engineer import FeatureEngineering
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

    # Feature engineering
    # Assuming the dataset has a 'timestamp' column and a target column 'price_next_hour'
    feature_engineering = FeatureEngineering(lags=[1, 2, 3], rolling_windows=[3, 6, 24])
    df = feature_engineering.transform(df, target_column="price_next_hour")

    features = feature_engineering.get_feature_names(df)
    target = "price_next_hour"

    # Ensure the target variable is not in the features
    X = df[features]
    y = df[target]

    numeric = [
        col
        for col in X.columns
        if X[col].dtype in [np.float64, np.int64] and col != "is_weekend"
    ]
    binary = ["is_weekend"]
    categorical = []  # or ["day_of_week"], and so on

    model_type = "rf"  # gb = gradient boosting, rf = ranfom forest to*do: pass later via CLI/config

    # Build the pipeline and train the model
    pipeline = build_pipeline(
        numeric_features=numeric,
        binary_features=binary,
        categorical_features=categorical,
        model_type=model_type,
    )

    # Split the data into training and testing sets
    # Using shuffle=False to maintain the time series order
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )
    print(f"Training on {len(X_train)} samples, testing on {len(X_test)} samples")

    # Train the pipeline
    pipeline.fit(X_train, y_train)

    # Evaluate the model
    y_pred = pipeline.predict(X_test)
    print(f"MAE: {mean_absolute_error(y_test, y_pred):.2f}")
    print(f"R² : {r2_score(y_test, y_pred):.2f}")

    MODEL_PATH = get_model_path(model_type)
    joblib.dump(pipeline, MODEL_PATH)
    print(f"✅ Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    train()

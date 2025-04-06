from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer


def build_pipeline(numeric_features):
    """
    1. Build a machine learning pipeline with preprocessing and model training.
    2. The pipeline includes a StandardScaler for numeric features and a RandomForestRegressor.
    3. The pipeline is returned as a single object.
    4. The pipeline can be used for both training and prediction.
    5. The pipeline is designed to handle numeric features only.
    """

    numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
        ]
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", RandomForestRegressor(n_estimators=100, random_state=42)),
        ]
    )

    return pipeline

from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer


def build_pipeline(numeric_features, binary_features=None, categorical_features=None):
    """
    1. Build a machine learning pipeline with preprocessing and model training.
    2. The pipeline includes a StandardScaler for numeric features and a RandomForestRegressor.
    3. The pipeline is returned as a single object.
    4. The pipeline can be used for both training and prediction.
    5. The pipeline is designed to handle numeric features only.
    """

    numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])

    binary_transformer = "passthrough"

    categorical_transfomer = (
        OneHotEncoder(handle_unknown="ignore") if categorical_features else "drop"
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("binary", binary_transformer, binary_features, binary_features or []),
            (
                "cat",
                categorical_transfomer,
                categorical_features,
                categorical_features or [],
            ),
        ]
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", RandomForestRegressor(n_estimators=100, random_state=0)),
        ]
    )

    return pipeline

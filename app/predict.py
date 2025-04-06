import numpy as np
import joblib

model = joblib.load("app/rf_model.pkl")


def make_prediction(features: list[float]) -> float:
    """
    Make a prediction using the trained Random Forest model.

    Parameters:
    - features: List of feature values for prediction

    Returns:
    - Prediction result
    """
    # Ensure features are in the correct shape for the model
    features = np.array(features).reshape(1, -1)

    # Make prediction
    prediction = model.predict(features)

    return float(prediction[0])

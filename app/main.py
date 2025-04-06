from fastapi import FastAPI
from pydantic import BaseModel
from app.predict import make_prediction

app = FastAPI()


@app.get("/")
def read_root():
    """
    Root endpoint to check if the API is running.

    Returns:
    - Message indicating the API is running
    """
    return {"message": "API is running"}


class Features(BaseModel):
    features: list[float]


@app.post("/predict")
def predict(features: Features):
    """
    Endpoint to make predictions using the trained model.

    Parameters:
    - features: List of feature values for prediction

    Returns:
    - Prediction result
    """
    prediction = make_prediction(features.features)
    return {"prediction": prediction}

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib


def train_model(X_train, y_train):
    """
    Train a Random Forest model on the provided training data.
    
    Parameters:
    - X_train: Features for training
    - y_train: Target variable for training
    
    Returns:
    - model: Trained Random Forest model
    """
    model = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
    model.fit(X_train, y_train)

    return joblib.dump(model, "app/rf_model.pkl")




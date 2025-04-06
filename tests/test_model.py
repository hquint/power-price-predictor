from app.model import train_model
import os
import pandas as pd

def test_model_saves_file():
    df = pd.read_csv("data/power_market_data.csv")
    X = df.drop(columns=["price_next_hour"])
    y = df["price_next_hour"]
    
    train_model(X,y)
    
    assert os.path.exists("app/rf_model.pkl")
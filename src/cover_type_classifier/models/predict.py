import pandas as pd
import numpy as np
import joblib


def load_model(model_path: str):
    """Loads model from binary file."""
    with open(model_path, "rb") as file:
        model, pipeline = joblib.load(file)
    return model, pipeline


def predict(model_path: str, data: dict) -> np.ndarray:
    df = pd.DataFrame([data])
    df.columns = df.columns.str.lower()
    model, pipeline = load_model(model_path)
    df = pipeline.transform(df)
    prediction = model.predict(df)
    return prediction

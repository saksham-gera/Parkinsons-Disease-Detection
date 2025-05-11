from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np
import pandas as pd

# Load model and scaler
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Hardcoded feature names from Parkinson’s dataset
feature_names = [
    "MDVP:Fo(Hz)", "MDVP:Fhi(Hz)", "MDVP:Flo(Hz)", "MDVP:Jitter(%)", "MDVP:Jitter(Abs)",
    "MDVP:RAP", "MDVP:PPQ", "Jitter:DDP", "MDVP:Shimmer", "MDVP:Shimmer(dB)",
    "Shimmer:APQ3", "Shimmer:APQ5", "MDVP:APQ", "Shimmer:DDA", "NHR", "HNR",
    "RPDE", "DFA", "spread1", "spread2", "D2", "PPE"
]

app = FastAPI()

class InputData(BaseModel):
    input_data: dict

@app.post("/predict")
def predict(data: InputData):
    input_dict = data.input_data
    # Fill in missing features with 0
    final_input = [input_dict.get(f, 0) for f in feature_names]
    scaled_input = scaler.transform([final_input])
    prediction = model.predict(scaled_input)
    return {"prediction": int(prediction[0])}

from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# Initialize FastAPI app
app = FastAPI()

# Load the trained model
model = joblib.load("regression_model.pkl")

# Define input data schema
class InputData(BaseModel):
    feature: float

# Prediction endpoint
@app.post("/predict")
def predict(data: InputData):
    input_data = np.array([[data.feature]])
    prediction = model.predict(input_data)
    return {"prediction": prediction[0]}

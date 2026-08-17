from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
from pathlib import Path

app = FastAPI(title="Maternal Health Risk Prediction API")

# Load the saved model and label encoder once, when the API starts
PROJECT_ROOT = Path(__file__).resolve().parent.parent
model = joblib.load(PROJECT_ROOT / "models" / "maternal_risk_model.pkl")
le = joblib.load(PROJECT_ROOT / "models" / "label_encoder.pkl")

# Define the shape of an incoming request - this is what the app will send
class PatientVitals(BaseModel):
    Age: float
    SystolicBP: float
    DiastolicBP: float
    BS: float
    BodyTemp: float
    HeartRate: float

@app.get("/")
def root():
    return {"message": "Maternal Health Risk Prediction API is running"}

@app.post("/predict")
def predict_risk(vitals: PatientVitals):
    # Convert incoming vitals into the same format the model expects
    input_df = pd.DataFrame([{
        "Age": vitals.Age,
        "SystolicBP": vitals.SystolicBP,
        "DiastolicBP": vitals.DiastolicBP,
        "BS": vitals.BS,
        "BodyTemp": vitals.BodyTemp,
        "HeartRate": vitals.HeartRate,
    }])

    # Predict and convert the numeric result back to a text label
    prediction_encoded = model.predict(input_df)[0]
    prediction_label = le.inverse_transform([prediction_encoded])[0]

    # Also get confidence scores for each class
    probabilities = model.predict_proba(input_df)[0]
    confidence = {le.classes_[i]: round(float(probabilities[i]), 3) for i in range(len(le.classes_))}

    return {
        "predicted_risk_level": prediction_label,
        "confidence_scores": confidence
    }

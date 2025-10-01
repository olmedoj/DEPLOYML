"""
FastAPI application for Diabetes prediction model inference.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict
import joblib
import numpy as np
from pathlib import Path

# Initialize FastAPI application
app = FastAPI(
    title="Diabetes Prediction API",
    description="API for predicting diabetes risk using machine learning",
    version="1.0.0"
)

# Global variables for model and encoder
pipeline = None
label_encoder = None

# Model file paths
PIPELINE_PATH = Path("diabetes_pipeline.pkl")
ENCODER_PATH = Path("label_encoder.pkl")


class PatientData(BaseModel):
    """Input schema for patient medical data."""
    preg: float = Field(..., ge=0, le=20)
    plas: float = Field(..., ge=0, le=300)
    pres: float = Field(..., ge=0, le=200)
    skin: float = Field(..., ge=0, le=100)
    insu: float = Field(..., ge=0, le=900)
    mass: float = Field(..., ge=0, le=70)
    pedi: float = Field(..., ge=0, le=3)
    age: float = Field(..., ge=21, le=120)

    class Config:
        schema_extra = {
            "example": {
                "preg": 6, "plas": 148, "pres": 72, "skin": 35,
                "insu": 0, "mass": 33.6, "pedi": 0.627, "age": 50
            }
        }


class PredictionResponse(BaseModel):
    """Output schema for prediction results."""
    prediction: str
    probability_negative: float
    probability_positive: float
    confidence: float


@app.on_event("startup")
async def startup_event():
    """Load model artifacts when the API starts."""
    global pipeline, label_encoder
    pipeline = joblib.load(PIPELINE_PATH)
    label_encoder = joblib.load(ENCODER_PATH)


@app.get("/")
async def root():
    """Root endpoint with basic API information."""
    return {"message": "Diabetes Prediction API", "version": "1.0.0"}


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "model_loaded": pipeline is not None}


@app.post("/predict", response_model=PredictionResponse)
async def predict(patient_data: PatientData):
    """Make a diabetes prediction for a single patient."""
    if pipeline is None or label_encoder is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Convert input data to numpy array
    input_array = np.array([[
        patient_data.preg, patient_data.plas, patient_data.pres, patient_data.skin,
        patient_data.insu, patient_data.mass, patient_data.pedi, patient_data.age
    ]])
    
    # Make prediction
    prediction = pipeline.predict(input_array)
    prediction_proba = pipeline.predict_proba(input_array)
    
    # Prepare response
    return PredictionResponse(
        prediction=label_encoder.inverse_transform(prediction)[0],
        probability_negative=float(prediction_proba[0][0]),
        probability_positive=float(prediction_proba[0][1]),
        confidence=float(max(prediction_proba[0]))
    )


@app.post("/predict/batch", response_model=List[PredictionResponse])
async def predict_batch(patients: List[PatientData]):
    """Make diabetes predictions for multiple patients."""
    if pipeline is None or label_encoder is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if len(patients) > 100:
        raise HTTPException(status_code=400, detail="Maximum 100 patients per batch")
    
    # Convert all patient data to numpy array
    input_data = np.array([[
        p.preg, p.plas, p.pres, p.skin, p.insu, p.mass, p.pedi, p.age
    ] for p in patients])
    
    # Make predictions
    predictions = pipeline.predict(input_data)
    predictions_proba = pipeline.predict_proba(input_data)
    
    # Prepare responses
    return [
        PredictionResponse(
            prediction=label_encoder.inverse_transform([predictions[i]])[0],
            probability_negative=float(predictions_proba[i][0]),
            probability_positive=float(predictions_proba[i][1]),
            confidence=float(max(predictions_proba[i]))
        )
        for i in range(len(patients))
    ]


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict
import joblib
import numpy as np
from pathlib import Path

app = FastAPI(
    title="API de Diabetes",
    description="API para predecir la probabilidad de diabetes en pacientes",
    version="1.0.0"
)

pipeline = None
label_encoder = None

PIPELINE_PATH = "modelo_diabetes.pkl"
LABEL_ENCODER_PATH = "label_encoder_diabetes.pkl"

class PatientData(BaseModel):
    preg: float = Field(..., ge=0, le=20)
    plas: float = Field(..., ge=0, le=300)
    pres: float = Field(..., ge=0, le=200)
    skin: float = Field(..., ge=0, le=100)
    insu: float = Field(..., ge=0, le=900)
    mass: float = Field(..., ge=0, le=70)
    pedi: float = Field(..., ge=0, le=3)
    age: int = Field(..., ge=21, le=120)

class PredictionResponse(BaseModel):
    prediction: str
    probability_negative: float
    probability_positive: float
    confidence: float


@app.on_event("startup")
async def startup_event():
    global pipeline, label_encoder
    pipeline = joblib.load(PIPELINE_PATH)
    label_encoder = joblib.load(LABEL_ENCODER_PATH)

@app.get("/")
async def root():
    return {"message": "Bienvenido a la API de Diabetes", "endpoints": ["/health", "/predict"]}

@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": pipeline is not None}

@app.post("/predict", response_model=PredictionResponse)
async def predict(patient_data: PatientData):
    if pipeline is None or label_encoder is None:
        return HTTPException(status_code=503, detail="Modelo no cargado")
    
    input_array = np.array([
        patient_data.preg,
        patient_data.plas,
        patient_data.pres,
        patient_data.skin,
        patient_data.insu,
        patient_data.mass,
        patient_data.pedi,
        patient_data.age
    ]).reshape(1, -1)
    prediction = pipeline.predict(input_array)
    prediction_proba = pipeline.predict_proba(input_array)
    print(prediction_proba)

    return PredictionResponse(
        prediction=label_encoder.inverse_transform(prediction)[0],
        probability_negative=prediction_proba[0][0],
        probability_positive=prediction_proba[0][1],
        confidence=prediction_proba[0][0]
    )

@app.post("/predict/batch", response_model=List[PredictionResponse])
async def predict_batch(patient_data: List[PatientData]):
    if pipeline is None or label_encoder is None:
        return HTTPException(status_code=503, detail="Modelo no cargado")
    if len(patient_data) > 100:
        return HTTPException(status_code=400, detail="Batch size must be less than or equal to 100")
    input_data = np.array(
        [(p.preg, p.plas, p.pres, p.skin, p.insu, p.mass, p.pedi, p.age) for p in patient_data]
    )
    print(input_data)
    predictions = pipeline.predict(input_data)
    prediction_proba = pipeline.predict_proba(input_data)

    return [
        PredictionResponse(
            prediction=label_encoder.inverse_transform(predictions)[i][0],
            probability_negative=prediction_proba[i][0],
            probability_positive=prediction_proba[i][1],
            confidence=prediction_proba[i][0]
        ) for i in range(len(patient_data))
    ]

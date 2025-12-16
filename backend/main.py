from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import Literal
import joblib
import os
import pandas as pd

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "models",
    "stroke_pipeline.joblib",
)

app = FastAPI(title="Stroke Prediction API")

origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = joblib.load(MODEL_PATH)


class Patient(BaseModel):
    gender: Literal["Male", "Female", "Other"]
    age: float = Field(..., ge=0, le=120)
    hypertension: int = Field(..., ge=0, le=1)
    heart_disease: int = Field(..., ge=0, le=1)
    ever_married: Literal["Yes", "No"]
    work_type: str
    Residence_type: Literal["Urban", "Rural"]
    avg_glucose_level: float = Field(..., ge=0, le=400)
    bmi: float = Field(..., gt=0, le=80)
    smoking_status: str

    @validator("work_type")
    def validate_work_type(cls, v: str) -> str:
        allowed = {"Private", "Self-employed", "Govt_job", "children", "Never_worked"}
        if v not in allowed:
            raise ValueError(f"work_type must be one of {allowed}")
        return v

    @validator("smoking_status")
    def validate_smoking(cls, v: str) -> str:
        allowed = {"never smoked", "formerly smoked", "smokes", "Unknown"}
        if v not in allowed:
            raise ValueError(f"smoking_status must be one of {allowed}")
        return v

@app.get("/")
def root():
    return {"message": "Stroke prediction API is running"}


@app.post("/predict")
def predict(patient: Patient):
    try:
        data = pd.DataFrame([patient.dict()])

        data["hypertension"] = data["hypertension"].astype(str)
        data["heart_disease"] = data["heart_disease"].astype(str)

        proba = model.predict_proba(data)[0, 1]
        pred = int(proba >= 0.5)

        return {"prediction": pred, "probability": float(proba)}
    except ValueError as e:
        # something off with input values
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        # generic failure (model / data shape etc.)
        raise HTTPException(
            status_code=500,
            detail="Internal model error. Please try again later.",
        )

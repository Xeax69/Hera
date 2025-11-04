from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
from pathlib import Path
from typing import Optional

app = FastAPI(title="HERA API", version="0.1.0")


# ---------- Chargement des modèles ----------
BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = (BASE_DIR / ".." / ".." / "src" / "models" / "xgb_climbing.joblib").resolve()
SCALER_PATH = (BASE_DIR / ".." / ".." / "src" / "models" / "scaler_climbing.joblib").resolve()

def load_joblib_model(path: Path):
    if not path.exists():
        print(f"[ERROR] Missing file: {path}")
        return None
    try:
        return joblib.load(path)
    except Exception as e:
        print(f"[ERROR] Unable to load {path}: {e}")
        return None

model = load_joblib_model(MODEL_PATH)
scaler = load_joblib_model(SCALER_PATH)


# ---------- SCHÉMAS ----------
class ClimbingFeatures(BaseModel):
    age: Optional[float] = None
    weight: Optional[float] = None
    hang_time_s: float
    route_grade: str


class ClimbingPrediction(BaseModel):
    performance_score: float
    advice: str
    model_version: str


@app.get("/health")
def health():
    status = "ok"
    if model is None or scaler is None:
        status = "warning: model or scaler not loaded"
    return {"status": status}


# ---------- PREDICTION HANG TIME ----------
@app.post("/predict/climbing/v1", response_model=ClimbingPrediction)
def predict(payload: ClimbingFeatures):

    score = max(0.0, min(1.0, payload.hang_time_s / 20.0))

    advice = (
        "Augmente progressivement les suspensions (5x10s, 2 séances/sem)."
        if score < 0.6 else
        "Consolide: maintien + technique de pose de pieds."
    )

    return {
        "performance_score": float(score),
        "advice": advice,
        "model_version": "0.0.1",
    }


# ---------- PREDICTION GRADE ----------
class ClimbingGradeInput(BaseModel):
    sex: int
    height: float
    weight: float
    age: float
    years_cl: int
    grades_count: int
    grades_first: float
    grades_last: float
    grades_max: float
    grades_mean: float

    # One-hot encoding
    country_AUS: int = 0
    country_AUT: int = 0
    country_BEL: int = 0
    country_BRA: int = 0
    country_CAN: int = 0
    country_CHE: int = 0
    country_CZE: int = 0
    country_DEU: int = 0
    country_DNK: int = 0
    country_ESP: int = 0
    country_FIN: int = 0
    country_FRA: int = 1
    country_GBR: int = 0
    country_HRV: int = 0
    country_ITA: int = 0
    country_MEX: int = 0
    country_NLD: int = 0
    country_NOR: int = 0
    country_POL: int = 0
    country_PRT: int = 0
    country_RUS: int = 0
    country_SVN: int = 0
    country_SWE: int = 0
    country_USA: int = 0
    country_ZAF: int = 0
    country_other: int = 0


class GradePrediction(BaseModel):
    predicted_grade: float
    model_version: str


@app.post("/predict/climbing-grade/v1", response_model=GradePrediction)
def predict_climbing_grade(payload: ClimbingGradeInput):

    if model is None:
        raise HTTPException(
            status_code=500,
            detail=f"Model missing or unreadable at {MODEL_PATH}"
        )

    if scaler is None:
        raise HTTPException(
            status_code=500,
            detail=f"Scaler missing or unreadable at {SCALER_PATH}"
        )

    # Transformer input → DataFrame
    input_df = pd.DataFrame([payload.dict()])

    try:
        input_scaled = scaler.transform(input_df)
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Scaler transform failed: {e}"
        )

    try:
        prediction = model.predict(input_scaled)[0]
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Model prediction failed: {e}"
        )

    return {
        "predicted_grade": float(prediction),
        "model_version": "xgb_climbing_1.0.0"
    }
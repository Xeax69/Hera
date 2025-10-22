from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="HERA API", version="0.1.0")


class ClimbingFeatures(BaseModel):
    age: float | None = None
    weight: float | None = None
    hang_time_s: float
    route_grade: str


class ClimbingPrediction(BaseModel):
    performance_score: float
    advice: str
    model_version: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict/climbing/v1", response_model=ClimbingPrediction)
def predict(payload: ClimbingFeatures):
    # Baseline FICTIVE en attendant un vrai modèle
    score = max(0.0, min(1.0, payload.hang_time_s / 20.0))
    advice = (
        "Augmente progressivement les suspensions (5x10s, 2 séances/sem)."
        if score < 0.6
        else "Consolide: maintien + technique de pose de pieds."
    )
    return {
        "performance_score": float(score),
        "advice": advice,
        "model_version": "0.0.1",
    }

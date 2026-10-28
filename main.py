from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

MODEL_PATH = Path(__file__).parent / "model.joblib"

app = FastAPI(title="Iris Predictor")
model = joblib.load(MODEL_PATH)


class IrisInput(BaseModel):
    SepalLengthCm: float
    SepalWidthCm: float
    PetalLengthCm: float
    PetalWidthCm: float


@app.get("/")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(data: IrisInput):
    X = pd.DataFrame([data.model_dump()])
    return {"prediction": str(model.predict(X)[0])}
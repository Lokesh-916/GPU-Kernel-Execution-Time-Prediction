from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import numpy as np
import joblib
from pathlib import Path
import math

# Paths to model artifacts (placed at repo root per user's message)
ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "random_forest_model.pkl"
SCALER_PATH = ROOT / "minmax_scaler.pkl"

# Load artifacts at startup
try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    raise RuntimeError(f"Failed to load model at {MODEL_PATH}: {e}")

try:
    scaler = joblib.load(SCALER_PATH)
except Exception as e:
    raise RuntimeError(f"Failed to load scaler at {SCALER_PATH}: {e}")

# API
app = FastAPI(title="SGEMM Predictor API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictRequest(BaseModel):
    # First 10 ordinal attributes (scaled by provided scaler)
    MWG: int = Field(...)
    NWG: int = Field(...)
    KWG: int = Field(...)
    MDIMC: int = Field(...)
    NDIMC: int = Field(...)
    MDIMA: int = Field(...)
    NDIMB: int = Field(...)
    KWI: int = Field(...)
    VWM: int = Field(...)
    VWN: int = Field(...)
    # Remaining 4 binary attributes (not scaled)
    STRM: int = Field(..., ge=0, le=1)
    STRN: int = Field(..., ge=0, le=1)
    SA: int = Field(..., ge=0, le=1)
    SB: int = Field(..., ge=0, le=1)


class PredictResponse(BaseModel):
    predicted_time_ms: float


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    try:
        # Order matters: first 10 go to scaler
        first10 = np.array([
            req.MWG, req.NWG, req.KWG, req.MDIMC, req.NDIMC,
            req.MDIMA, req.NDIMB, req.KWI, req.VWM, req.VWN
        ], dtype=float).reshape(1, -1)

        # Scale the first 10 attributes
        try:
            first10_scaled = scaler.transform(first10)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Scaler transform failed: {e}")

        # Append remaining 4 binary attributes as-is
        last4 = np.array([[req.STRM, req.STRN, req.SA, req.SB]], dtype=float)
        X = np.concatenate([first10_scaled, last4], axis=1)

        # Model inference: output is log1p(actual_ms)
        y_log = model.predict(X)
        # Convert back to ms: ms = expm1(y_log)
        y_ms = float(np.expm1(y_log[0]))
        # Guard against negative after numeric noise
        y_ms = max(y_ms, 0.0)
        return PredictResponse(predicted_time_ms=round(y_ms, 5))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")


@app.get("/health")
def health():
    return {"status": "ok"}

"""FastAPI service for MP134: predicts the auto-rickshaw fare.

Run locally from the repo root:
    PYTHONPATH=src uvicorn app.main:app --port 8000
"""
import json
import logging
import sys
import time
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field

from autofare.clean import clean_booking

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "model.joblib"


# --- JSON logging: one JSON line per event, written to stdout -------------
class JsonFormatter(logging.Formatter):
    def format(self, record):
        entry = {
            "time": self.formatTime(record),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        entry.update(getattr(record, "fields", {}))
        return json.dumps(entry)


logger = logging.getLogger("mp134")
_handler = logging.StreamHandler(sys.stdout)
_handler.setFormatter(JsonFormatter())
logger.handlers = [_handler]
logger.setLevel(logging.INFO)
logger.propagate = False


# --- Request schema: the client's two rules live here ---------------------
class RideRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")   # exactly these fields

    distance_km: float = Field(..., ge=0)    # rule 2: below 0 is a typing mistake
    wait_min: float = Field(..., le=55)      # rule 1: above 55 is impossible
    passengers: int
    hour: int
    luggage_bags: int
    fuel_price_rs: float
    booking: str
    zone: str


app = FastAPI(title="MP134 Auto-Rickshaw Fare (Patna)")
model = joblib.load(MODEL_PATH)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(ride: RideRequest):
    start = time.perf_counter()
    row = pd.DataFrame([ride.model_dump()])
    row["booking"] = clean_booking(row["booking"])   # "APP" -> "app"
    fare = round(float(model.predict(row)[0]), 2)
    logger.info("prediction", extra={"fields": {
        "event": "predict",
        "zone": ride.zone,
        "booking": row["booking"].iloc[0],
        "distance_km": ride.distance_km,
        "fare_rs": fare,
        "latency_ms": round((time.perf_counter() - start) * 1000, 2),
    }})
    return {"fare_rs": fare}
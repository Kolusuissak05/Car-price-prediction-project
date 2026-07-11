import json
import pickle
from pathlib import Path

import pandas as pd
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent

app = FastAPI(title="Car Price Predictor")

# ---- Load trained model + metadata at startup ----
with open(BASE_DIR / "model" / "car_price_model.pkl", "rb") as f:
    model = pickle.load(f)

with open(BASE_DIR / "model" / "metadata.json") as f:
    METADATA = json.load(f)


class CarInput(BaseModel):
    brand: str
    year: int = Field(..., ge=1980, le=2025)
    km_driven: float = Field(..., ge=0)
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: float = Field(..., ge=0)
    engine: float = Field(..., ge=0)
    max_power: float = Field(..., ge=0)
    seats: int = Field(..., ge=2, le=14)


@app.get("/", response_class=HTMLResponse)
def index():
    return HTMLResponse((BASE_DIR / "templates" / "index.html").read_text())


@app.post("/predict")
def predict(car: CarInput):
    if car.brand not in METADATA["brand_to_code"]:
        return {"error": f"Unknown brand: {car.brand}"}
    if car.fuel not in METADATA["fuel_map"]:
        return {"error": f"Unknown fuel type: {car.fuel}"}
    if car.seller_type not in METADATA["seller_map"]:
        return {"error": f"Unknown seller type: {car.seller_type}"}
    if car.transmission not in METADATA["transmission_map"]:
        return {"error": f"Unknown transmission: {car.transmission}"}
    if car.owner not in METADATA["owner_map"]:
        return {"error": f"Unknown owner type: {car.owner}"}

    row = {
        "name": METADATA["brand_to_code"][car.brand],
        "year": car.year,
        "km_driven": car.km_driven,
        "fuel": METADATA["fuel_map"][car.fuel],
        "seller_type": METADATA["seller_map"][car.seller_type],
        "transmission": METADATA["transmission_map"][car.transmission],
        "owner": METADATA["owner_map"][car.owner],
        "mileage": car.mileage,
        "engine": car.engine,
        "max_power": car.max_power,
        "seats": car.seats,
    }

    df = pd.DataFrame([row], columns=METADATA["feature_order"])
    prediction = float(model.predict(df)[0])
    prediction = max(prediction, 0.0)  # never show a negative price

    return {
        "predicted_price": round(prediction, 2),
        "predicted_price_formatted": f"₹{prediction:,.0f}",
    }


@app.get("/api/metadata")
def get_metadata():
    return {
        "brands": METADATA["brands"],
        "fuels": list(METADATA["fuel_map"].keys()),
        "sellers": list(METADATA["seller_map"].keys()),
        "transmissions": list(METADATA["transmission_map"].keys()),
        "owners": list(METADATA["owner_map"].keys()),
        "year_min": METADATA["year_min"],
        "year_max": METADATA["year_max"],
        "r2_score": METADATA["r2_score"],
        "mae": METADATA["mae"],
        "models_by_brand": METADATA["models_by_brand"],
        "model_defaults": METADATA["model_defaults"],
    }

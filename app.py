"""
Battery Early Warning API
Serves failure-risk predictions from the trained neural network.

Run locally:    uvicorn app:app --reload --port 8000
Docs:           http://localhost:8000/docs
"""

import json
import joblib
import torch
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import Optional


app = FastAPI(
    title="Voltrix Battery Early Warning API",
    description="""
    ### ⚡ AI-Powered Diagnostic Platform
    
    Predicts EV battery failure risks using real-time sensor telemetry.
    
    * **`/predict`**: Submit JSON feature readings to get failure predictions.
    * **`/health`**: Check system operational status.
    """,
    version="1.0.0",
    docs_url="/docs",      # Interactive Swagger UI
    redoc_url="/redoc"     # Alternative clean documentation UI
)

@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Voltrix Battery Predictor</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 600px; margin: 40px auto; padding: 20px; background-color: #f4f4f9; }
            h1 { text-align: center; color: #333; }
            .form-group { margin-bottom: 15px; }
            label { display: block; margin-bottom: 5px; font-weight: bold; }
            input { width: 100%; padding: 8px; box-sizing: border-box; border: 1px solid #ccc; border-radius: 4px; }
            button { width: 100%; padding: 10px; background-color: #007bff; color: white; border: none; border-radius: 4px; font-size: 16px; cursor: pointer; }
            button:hover { background-color: #0056b3; }
            #result { margin-top: 20px; padding: 15px; border-radius: 4px; display: none; background-color: #e9ecef; }
        </style>
    </head>
    <body>
        <h1>⚡ Voltrix Battery Failure Predictor</h1>
        <form id="predictionForm">
            <div class="form-group">
                <label for="vehicle_id">Vehicle ID</label>
                <input type="text" id="vehicle_id" name="vehicle_id" value="EV-101" required>
            </div>
            <div class="form-group">
                <label for="cycle_count">Cycle Count</label>
                <input type="number" id="cycle_count" name="cycle_count" value="100" required>
            </div>
            <div class="form-group">
                <label for="battery_health">Battery Health (%)</label>
                <input type="number" step="0.1" id="battery_health" name="battery_health" value="95.0" required>
            </div>
            <button type="button" onclick="makePrediction()">Predict</button>
        </form>

        <div id="result"></div>

        <script>
            async function makePrediction() {
                const vehicle_id = document.getElementById('vehicle_id').value;
                const cycle_count = parseFloat(document.getElementById('cycle_count').value);
                const battery_health = parseFloat(document.getElementById('battery_health').value);

                const payload = {
                    vehicle_id: vehicle_id,
                    readings: {
                        cycle_count: cycle_count,
                        battery_health_percent: battery_health
                    }
                };

                const resultDiv = document.getElementById('result');
                resultDiv.style.display = 'block';
                resultDiv.innerText = "Processing...";

                try {
                    const response = await fetch('/predict', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify(payload)
                    });
                    const data = await response.json();
                    resultDiv.innerText = "Result: " + JSON.stringify(data, null, 2);
                } catch (error) {
                    resultDiv.innerText = "Error: " + error.message;
                }
            }
        </script>
    </body>
    </html>
    """
# ---- Load artifacts once at startup ----
model = torch.jit.load("battery_failure_model.pt")
model.eval()
scaler = joblib.load("scaler.pkl")
with open("model_schema.json") as f:
    schema = json.load(f)

FEATURE_COLUMNS = schema["feature_columns"]
NUMERIC_COLS = schema["numeric_source_cols"]
CATEGORICAL_COLS = schema["categorical_source_cols"]
NUMERIC_MEDIANS = schema["numeric_medians"]
CAT_FILL = schema["categorical_fill_value"]
THRESHOLD = schema["decision_threshold"]


class BatteryReading(BaseModel):
    # Raw sensor payload — any subset of the original raw columns.
    # Missing fields are filled with training-time medians/"Unknown" automatically.
    vehicle_id: Optional[str] = None
    readings: dict  # e.g. {"cycle_count": 812, "battery_health_percent": 71.2, "vehicle_brand": "Tesla", ...}


class PredictionResponse(BaseModel):
    vehicle_id: Optional[str]
    failure_probability: float
    flagged_for_service: bool
    threshold_used: float


def preprocess(readings: dict) -> pd.DataFrame:
    row = {}
    for col in NUMERIC_COLS:
        row[col] = readings.get(col, NUMERIC_MEDIANS.get(col))
    for col in CATEGORICAL_COLS:
        row[col] = readings.get(col, CAT_FILL)

    df_row = pd.DataFrame([row])
    df_encoded = pd.get_dummies(df_row, columns=CATEGORICAL_COLS)

    # Align to the exact training-time schema: add any missing dummy columns as 0,
    # drop anything unexpected, and enforce column order.
    df_aligned = df_encoded.reindex(columns=FEATURE_COLUMNS, fill_value=0)
    return df_aligned


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: BatteryReading):
    try:
        X = preprocess(payload.readings)
        X_scaled = scaler.transform(X)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

        with torch.no_grad():
            logit = model(X_tensor)
            proba = torch.sigmoid(logit).item()

        return PredictionResponse(
            vehicle_id=payload.vehicle_id,
            failure_probability=round(proba, 4),
            flagged_for_service=proba >= THRESHOLD,
            threshold_used=THRESHOLD,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

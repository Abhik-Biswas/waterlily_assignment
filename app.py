"""FastAPI inference service for the saved diabetes readmission XGBoost model.

Run: uvicorn app:app --reload
"""
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict

MODEL_PATH = Path(__file__).with_name("xgboost_model.json")

CATEGORICAL_COLS = [
    "age", "weight_imputed", "race", "gender", "payer_code",
    "metformin", "repaglinide", "nateglinide", "chlorpropamide",
    "glimepiride", "acetohexamide", "glipizide", "glyburide",
    "tolbutamide", "pioglitazone", "rosiglitazone", "acarbose",
    "miglitol", "troglitazone", "tolazamide", "examide", "citoglipton",
    "insulin", "glyburide-metformin", "glipizide-metformin",
    "glimepiride-pioglitazone", "metformin-rosiglitazone",
    "metformin-pioglitazone", "change", "description_admissions_source",
    "description_discharge_disposition", "description_admission_type",
]

app = FastAPI(title="Diabetes Readmission Prediction API", version="1.0.0")
model = xgb.XGBClassifier(enable_categorical=True, tree_method="hist")
if not MODEL_PATH.exists():
    raise RuntimeError(f"Missing model file: {MODEL_PATH}. Run the notebook's model.save_model cell first.")
model.load_model(str(MODEL_PATH))
booster = model.get_booster()
FEATURES = booster.feature_names
if not FEATURES:
    raise RuntimeError("Saved model has no feature names; export the model trained on a pandas DataFrame.")
FEATURE_TYPES = dict(zip(FEATURES, booster.feature_types or []))


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="allow")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(payload: PredictionRequest):
    record: dict[str, Any] = payload.model_dump()
    missing = [col for col in FEATURES if col not in record]
    if missing:
        raise HTTPException(status_code=422, detail={"missing_features": missing})

    row = pd.DataFrame([{col: record[col] for col in FEATURES}], columns=FEATURES)
    for col in FEATURES:
        if FEATURE_TYPES.get(col) == "c" or col in CATEGORICAL_COLS:
            row[col] = row[col].astype("category")
        else:
            try:
                row[col] = pd.to_numeric(row[col], errors="raise")
            except (ValueError, TypeError) as exc:
                raise HTTPException(status_code=422, detail=f"Invalid numeric value for {col}: {exc}") from exc

    try:
        probabilities = model.predict_proba(row)[0]
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Model inference failed: {exc}") from exc

    # LabelEncoder fitted on ['<30', '>30', 'NO'] sorts labels alphabetically.
    # Verify against le.classes_ in the notebook before relying on this mapping.
    class_labels = ["<30", ">30", "NO"]
    prediction_idx = int(np.argmax(probabilities))
    return {
        "prediction": class_labels[prediction_idx],
        "prediction_code": prediction_idx,
        "probabilities": {
            label: float(probabilities[i]) for i, label in enumerate(class_labels)
        },
    }

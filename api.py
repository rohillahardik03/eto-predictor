from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
import numpy as np
import pickle

app = FastAPI()

# Load models at startup
with open('eto_ann_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('scaler_X.pkl', 'rb') as f:
    scaler_X = pickle.load(f)
with open('scaler_y.pkl', 'rb') as f:
    scaler_y = pickle.load(f)
with open('feature_names.pkl', 'rb') as f:
    feature_names = pickle.load(f)
with open('smart_imputer.pkl', 'rb') as f:
    smart_imputer = pickle.load(f)

class EToPredictionRequest(BaseModel):
    n: Optional[float] = None       # Sunshine hours
    tmax: Optional[float] = None    # Max temperature
    tmin: Optional[float] = None    # Min temperature
    rhmax: Optional[float] = None   # Max relative humidity
    rhmin: Optional[float] = None   # Min relative humidity
    u: Optional[float] = None       # Wind speed

@app.post("/predict")
def predict(request: EToPredictionRequest):
    input_map = {
        'n': request.n,
        'Tmax (°C)': request.tmax,
        'Tmin (°C)': request.tmin,
        'RHmax': request.rhmax,
        'RHmin': request.rhmin,
        'u ': request.u
    }
    provided = sum(1 for v in input_map.values() if v is not None)
    if provided < 2:
        return {"error": "At least 2 parameters required"}

    input_array = np.full(len(feature_names), np.nan)
    for i, feat in enumerate(feature_names):
        if input_map.get(feat) is not None:
            input_array[i] = input_map[feat]

    imputed = smart_imputer.transform(input_array.reshape(1, -1))
    scaled = scaler_X.transform(imputed)
    pred_scaled = model.predict(scaled)
    pred = scaler_y.inverse_transform(pred_scaled.reshape(-1, 1)).ravel()[0]
    imputed_map = {feature_names[i]: round(float(imputed[0][i]), 3) for i in range(len(feature_names))}

    return {
        "eto": round(float(pred), 3),
        "unit": "mm/day",
        "params_provided": provided,
        "imputed_values": imputed_map
    }

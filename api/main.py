from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import joblib
import pickle
import os

app = FastAPI(title="Drug Response Prediction API")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

model_path = os.path.join(BASE_DIR, "models", "xgboost_model.joblib")
scaler_path = os.path.join(BASE_DIR, "data", "processed", "scaler.pkl")
genes_path = os.path.join(BASE_DIR, "data", "processed", "selected_genes.pkl")

model = joblib.load(model_path)

with open(scaler_path, "rb") as f:
    scaler = pickle.load(f)

with open(genes_path, "rb") as f:
    selected_genes = pickle.load(f)

class PredictionRequest(BaseModel):
    gene_expression: list[float]

class PredictionResponse(BaseModel):
    predicted_ic50: float
    sensitivity: str

@app.get("/")
def root():
    return {"message": "Drug Response Prediction API", "docs": "/docs"}

@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": True, "n_genes": len(selected_genes)}

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    if len(request.gene_expression) != len(selected_genes):
        raise HTTPException(
            status_code=400,
            detail=f"Expected {len(selected_genes)} genes, got {len(request.gene_expression)}"
        )
    
    features = np.array(request.gene_expression).reshape(1, -1)
    features_scaled = scaler.transform(features)
    
    prediction = model.predict(features_scaled)[0]
    
    sensitivity = "sensitive" if prediction < 2 else "resistant"
    
    return PredictionResponse(
        predicted_ic50=round(float(prediction), 4),
        sensitivity=sensitivity
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
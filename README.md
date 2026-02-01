# Drug Response Prediction Platform

Predicting cancer drug sensitivity from gene expression profiles using machine learning.

## Problem

Different cancer patients respond differently to the same drug. This tool predicts how sensitive a cancer cell will be to specific drugs based on its gene expression profile.

## Data

- **GDSC**: Drug response data (IC50) for 286 drugs across 969 cell lines
- **DepMap**: Gene expression data for 19,215 genes across 1,699 cell lines
- **Matched samples**: 714 cell lines with both drug response and expression data

## Models

| Model | R² Score |
|-------|----------|
| XGBoost | 0.208 |
| PyTorch MLP | 0.171 |

## Features

- Feature selection using variance filtering (19k → 1k genes)
- XGBoost and PyTorch model comparison
- SHAP interpretability for gene importance
- FastAPI prediction service
- Docker containerization

## Project Structure
```
drug-response-prediction/
├── notebooks/
│   ├── 01_exploration.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_xgboost_model.ipynb
│   ├── 04_pytorch_model.ipynb
│   └── 05_shap_interpretability.ipynb
├── api/
│   └── main.py
├── models/
├── data/
│   ├── raw/
│   └── processed/
├── reports/
│   └── figures/
├── Dockerfile
└── requirements.txt
```

## Installation
```bash
git clone https://github.com/YOUR_USERNAME/drug-response-prediction.git
cd drug-response-prediction
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Run API Locally
```bash
cd api
python main.py
```

API available at `http://localhost:8000`

## Run with Docker
```bash
docker build -t drug-response-api .
docker run -p 8000:8000 drug-response-api
```

## API Endpoints

- `GET /health` - Check API status
- `POST /predict` - Predict IC50 from gene expression

## Example Request
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"gene_expression": [0.5, 0.5, ...]}'  # 1000 values
```

## Technologies

- Python, Pandas, NumPy
- XGBoost, PyTorch
- SHAP
- FastAPI
- Docker
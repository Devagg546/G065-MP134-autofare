# G065 – MP134: Auto-Rickshaw Fare (Patna)

SCSE3040 MLOps mini project (Batch 14, Faculty: Mr. Dhananjay Kumar).

**Group ID:** G065
**Problem ID:** MP134
**Client:** Patna Auto Drivers' Cooperative (a made-up organisation)

## Members

| Enrolment No. | Name |
|---|---|
| S24CSEU0846 | Dakshesh Goel |
| S24CSEU1144 | Vishal Kumar |
| S24CSEU0832 | Dev Aggarwal |
| S24CSEU1142 | Priyanshu Raj |

## Problem

Predict `fare_rs`, the fare of an auto-rickshaw ride in rupees. This is a
regression problem measured with MAE (Mean Absolute Error).

## Project structure

- `src/autofare/` – data checks, cleaning, training and monitoring code
- `app/main.py` – FastAPI service (`POST /predict`, `GET /health`)
- `tests/` – pytest tests
- `data/raw/` – DVC-tracked data (the CSVs are not in git)
- `models/` – the champion model used by the service
- `k8s/` – Kubernetes Deployment and Service files
- `reports/findings.yaml` – data problems and monitoring results
- `predictions/` – holdout predictions

## How to run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest
uvicorn app.main:app --port 8000
```

## Status

Work in progress.

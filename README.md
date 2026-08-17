# Maternal Health Risk Prediction

A machine-learning project for predicting maternal health risk levels from patient vital-sign data.

## Project structure

```text
maternal-health-risk-prediction/
├── api/                 # FastAPI application
├── src/                 # Cleaning, analysis, visualization, and model-saving scripts
│   └── models/          # Model-training scripts
├── charts/              # Generated charts
├── models/              # Saved model and label encoder
├── docs/                # Project documentation
├── requirements.txt
├── .gitignore
└── README.md
```

## Setup

```bash
python -m venv .venv
.venv\\Scripts\\activate
pip install -r requirements.txt
```

## Run the API

From the project root:

```bash
uvicorn api.api:app --reload
```

Visit `http://127.0.0.1:8000/docs` to test the prediction endpoint.

## Data files

The original CSV datasets are not included because the requested GitHub layout does not contain a `data/` directory. To rerun the data-processing or training scripts, keep the CSV files locally and update their input paths as appropriate.

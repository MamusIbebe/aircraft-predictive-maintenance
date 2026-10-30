# Aircraft Engine Predictive Maintenance

Predicts how many flights a jet engine has left before it needs maintenance
(remaining useful life, RUL) from its sensor data, and turns that into a ranked
maintenance list and a fleet readiness number.

Data: NASA C-MAPSS turbofan engine dataset (FD001): 100 engines run to failure for
training, 100 in-service engines for testing, 21 sensors per flight.

## Results (NASA test set, 100 engines)

| Model | Test RMSE (flights) | Test MAE (flights) |
|---|---|---|
| Always guess the average | about 42 | n/a |
| Gradient boosting (scikit-learn) | 17.2 | 12.2 |
| **LSTM (PyTorch), deployed** | **14.5** | **10.7** |

## How it works

1. `src/data.py`: load, validate (missing values, duplicates, gaps) and label RUL, capped at 125
2. `src/split.py`: split by engine (80 train / 20 validation) to avoid data leakage
3. `src/features.py`: drop 7 flat sensors, standardize on training data only, rolling 5/10/20-flight stats
4. `src/sql_analysis.py`: DuckDB SQL (GROUP BY, CTE, window functions)
5. `src/train_baseline.py`: gradient boosting baseline; RMSE, MAE, NASA asymmetric score
6. `src/train_lstm.py`: PyTorch LSTM on 30-flight windows, dropout, early stopping
7. `src/experiments.py`: MLflow experiment tracking and hyperparameter comparison
8. `tests/` + `src/quality_gate.py`: unit tests and a gate that fails CI if test RMSE > 22
9. `src/api.py`: FastAPI service (`/health`, `/predict`, `/predict/fleet-engine/{id}`)
10. `Dockerfile`: CPU-only, non-root container image for the API
11. `.github/workflows/ml-pipeline.yml`: test, download data, train, gate, drift check, build and smoke-test the image
12. `src/drift.py`: PSI drift monitoring per sensor (with a simulated sensor fault)
13. `src/dashboard.py`: Streamlit fleet readiness dashboard

## Run it

```
python -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
# download data into data/raw (see the workflow file), then:
python src/data.py && python src/split.py && python src/features.py
python src/train_lstm.py && python src/quality_gate.py
uvicorn api:app --app-dir src --port 8000        # API docs at http://127.0.0.1:8000/docs
streamlit run src/dashboard.py                    # dashboard at http://127.0.0.1:8501
```

## Next steps

- Prediction ranges (how sure the model is), e.g. quantile regression or MC dropout
- Multiple operating conditions (FD002/FD004)
- Deploy the image to EKS/ECS with Argo CD; MLflow on RDS + S3; scheduled drift job with alerts

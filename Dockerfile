# Step 13: package the prediction API into a container image.
FROM docker.io/library/python:3.12-slim

WORKDIR /app

# CPU-only PyTorch (~200 MB) instead of the default GPU build (~3 GB).
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir pandas numpy scikit-learn joblib fastapi uvicorn

# Copy only what the API needs.
COPY src/ src/
COPY models/feature_prep.joblib models/lstm.pt models/
COPY data/raw/test_FD001.txt data/raw/

# Don't run as root inside the container.
RUN useradd --create-home appuser
USER appuser

EXPOSE 8000
CMD ["uvicorn", "api:app", "--app-dir", "src", "--host", "0.0.0.0", "--port", "8000"]

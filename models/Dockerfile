# syntax=docker/dockerfile:1

# GPU-enabled option (uncomment to use):
# FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
COPY api/requirements.txt /app/api-requirements.txt

RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install --upgrade pip \
    && /opt/venv/bin/pip install --no-cache-dir -r /app/requirements.txt -r /app/api-requirements.txt \
    && /opt/venv/bin/pip install --no-cache-dir xgboost lightgbm scikit-learn optuna

ENV PATH="/opt/venv/bin:$PATH"

COPY models/ /app/models/
COPY scripts/ /app/scripts/

VOLUME ["/app/model-registry", "/app/training-data"]

ENTRYPOINT ["python", "/app/scripts/train_all.py"]

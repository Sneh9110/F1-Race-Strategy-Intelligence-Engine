# syntax=docker/dockerfile:1

FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /build

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /build/requirements.txt
COPY api/requirements.txt /build/api-requirements.txt

RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install --upgrade pip \
    && /opt/venv/bin/pip install --no-cache-dir -r /build/requirements.txt -r /build/api-requirements.txt


FROM python:3.11-slim AS runtime

LABEL org.opencontainers.image.title="F1 Strategy Intelligence API" \
      org.opencontainers.image.source="https://github.com/Sneh9110/F1-Race-Strategy-Intelligence-Engine" \
      org.opencontainers.image.description="FastAPI backend for F1 Race Strategy Intelligence Engine"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH"

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd -m -u 10001 appuser

COPY --from=builder /opt/venv /opt/venv

COPY api/ /app/api/
COPY app/ /app/app/
COPY config/ /app/config/
COPY decision_engine/ /app/decision_engine/
COPY simulation/ /app/simulation/
COPY models/ /app/models/
COPY data_pipeline/ /app/data_pipeline/
COPY pyproject.toml /app/pyproject.toml

RUN mkdir -p /app/logs \
    && chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
  CMD curl -fsS http://localhost:8000/api/v1/health/ready || exit 1

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]

# syntax=docker/dockerfile:1

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
COPY api/requirements.txt /app/api-requirements.txt

RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install --upgrade pip \
    && /opt/venv/bin/pip install --no-cache-dir -r /app/requirements.txt -r /app/api-requirements.txt \
    && /opt/venv/bin/pip install --no-cache-dir pandas numpy pyarrow

ENV PATH="/opt/venv/bin:$PATH"

COPY data_pipeline/ /app/data_pipeline/
COPY config/ /app/config/
COPY scripts/ /app/scripts/

RUN useradd -m -u 10002 worker \
    && chown -R worker:worker /app

USER worker

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import os,redis; r=redis.from_url(os.environ.get('REDIS_URL','redis://redis:6379/0')); r.ping(); print('ok')" || exit 1

ENTRYPOINT ["python", "/app/scripts/run_ingestion.py"]
CMD ["live"]

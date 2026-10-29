# Small official Python image matching the project version
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    POETRY_VERSION=2.5.1 \
    POETRY_VIRTUALENVS_CREATE=false \
    POETRY_NO_INTERACTION=1 \
    MLFLOW_TRACKING_URI=sqlite:////app/mlruns/mlflow.db

WORKDIR /app

# 1. Poetry itself (rarely changes)
RUN pip install --no-cache-dir "poetry==${POETRY_VERSION}"

# 2. Production dependencies only (cached until pyproject.toml or poetry.lock change)
COPY pyproject.toml poetry.lock README.md ./
RUN poetry install --only main --no-root

# 3. Project code and configuration (change often, so they come last)
COPY src ./src
COPY params.yaml ./
RUN poetry install --only main

# 4. Run as a non-root user, owning the folders the pipeline writes to
RUN useradd --create-home appuser \
    && mkdir -p data models metrics reports mlruns \
    && chown -R appuser:appuser /app
USER appuser

CMD ["python", "-m", "bank_propensity_pipeline.pipeline"]

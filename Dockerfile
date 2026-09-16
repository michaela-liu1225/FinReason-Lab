FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

ARG FINREASON_EXTRAS=api

COPY pyproject.toml README.md LICENSE ./
COPY finreason ./finreason

RUN python -m pip install --upgrade pip \
    && python -m pip install ".[${FINREASON_EXTRAS}]"

RUN useradd --create-home --uid 10001 appuser
USER appuser

EXPOSE 8000
CMD ["uvicorn", "finreason.api:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]

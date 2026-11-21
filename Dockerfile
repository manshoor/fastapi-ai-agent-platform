FROM ghcr.io/astral-sh/uv:0.13.6 AS uv

FROM python:3.13-slim-bookworm AS runtime
COPY --from=uv /uv /uvx /usr/local/bin/
ENV UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1
WORKDIR /app
# OCR for scanned PDF pages: pdftoppm renders a page, tesseract reads it. Pinned.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        tesseract-ocr=5.3.0-2 \
        tesseract-ocr-eng=1:4.1.0-2 \
        poppler-utils=22.12.0-2+deb12u3 \
    && rm -rf /var/lib/apt/lists/*
RUN useradd --create-home --uid 10001 app
COPY pyproject.toml uv.lock ./
COPY app ./app
COPY alembic.ini ./
COPY migrations ./migrations
RUN uv sync --locked --no-dev --no-editable \
    && chmod -R a+rX /opt/venv \
    && chown -R app:app /app
USER app
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--timeout-keep-alive", "2", "--timeout-graceful-shutdown", "15"]

FROM runtime AS test
USER root
COPY --chown=app:app tests ./tests
COPY --chown=app:app scripts ./scripts
COPY --chown=app:app observability/collector.verify.yaml ./observability/collector.verify.yaml
RUN uv sync --locked --no-editable \
    && chmod -R a+rX /opt/venv \
    && chown -R app:app /app
USER app
CMD ["pytest", "--cov=app", "--cov-report=term-missing"]

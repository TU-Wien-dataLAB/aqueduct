# Base Python image
FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Builder stage for dependencies
FROM base AS builder
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
ENV UV_SYSTEM_PYTHON=true

WORKDIR /app

# Copy dependency file for layer caching
COPY ./pyproject.toml /app/pyproject.toml

# Install Python dependencies
RUN uv pip install --prefix=/install ".[postgresql]"

# Final image
FROM base AS final

# Copy installed Python packages
COPY --from=builder /install /usr/local

# Copy app code
COPY . /app/

WORKDIR /app/aqueduct

EXPOSE 8000

# The timeout for httpx.get() (e.g., 4s) should be less than the HEALTHCHECK --timeout (5s)
# to allow httpx to handle the timeout gracefully.
HEALTHCHECK --interval=10s --timeout=5s --start-period=5s --retries=5 \
  CMD ["python", "-c", "import httpx; httpx.get('http://localhost:8000/health', timeout=4).raise_for_status()"]

CMD ["daphne", "-b", "0.0.0.0", "-p", "8000", "aqueduct.asgi:application"]

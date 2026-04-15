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
RUN uv pip install --prefix=/install --group=dev-locust

# Final image with dependencies for running locust tests
FROM base AS final

# Copy installed Python packages
COPY --from=builder /install /usr/local

# Create a user, so that the processes in the containers don't run as root
RUN groupadd -g 1000 mockapi && useradd -u 1000 -g 1000 -r mockapi

# Copy app code, set ownership
COPY ./aqueduct /app/aqueduct
RUN chown -R mockapi:mockapi /app

USER mockapi
WORKDIR /app/aqueduct

# `EXPOSE` is not necessary, but let's keep it for documentation's sake
EXPOSE 45999

CMD ["python", "-m", "mock_api.mock_server", "--port", "45999"]

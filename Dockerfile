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

# Create a user, so that the processes in the containers don't run as root
RUN groupadd -g 1000 aqueduct && useradd -u 1000 -g 1000 -r aqueduct

# Copy app code, set ownership
COPY . /app/
RUN chown -R aqueduct:aqueduct /app

USER aqueduct
WORKDIR /app/aqueduct

EXPOSE 8000

CMD ["daphne", "-b", "0.0.0.0", "-p", "8000", "aqueduct.asgi:application"]

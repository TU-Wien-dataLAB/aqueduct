FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

FROM base AS builder

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy pyproject.toml so pip install can be cached efficiently when the code changed but not the packages
COPY ./pyproject.toml /app/pyproject.toml

# Install Python dependencies (including postgres dependencies)
RUN pip install --no-cache --upgrade pip \
    && pip install --prefix=/install --no-cache ".[postgresql]"
RUN cp -r /install/* /usr/local

WORKDIR /app/aqueduct


FROM base

COPY --from=builder /install /usr/local

# Copy project files
COPY . /app/
WORKDIR /app/aqueduct

# Expose port for Daphne
EXPOSE 8000

# Use Daphne to serve the ASGI app
CMD ["daphne", "-b", "0.0.0.0", "-p", "8000", "aqueduct.asgi:application"]

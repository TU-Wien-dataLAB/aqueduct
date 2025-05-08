# Stage 1: Base image with Python
FROM python:3.12-slim AS base

# Set environment variables for Python
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
# Tell uv to use the system-provided Python, avoids uv creating its own venv implicitly
# when not using `uv venv` and installing with --prefix.
ENV UV_SYSTEM_PYTHON=true

# Stage 2: Builder stage for installing dependencies
FROM base AS builder
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Install system dependencies:
# - build-essential: For compiling Python packages with C extensions (e.g., some database drivers)
# - curl: For downloading the uv installer
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy only the pyproject.toml file first to leverage Docker's layer caching.
# If your project structure changes or dependencies in pyproject.toml change,
# this layer and subsequent layers will be rebuilt.
COPY ./pyproject.toml /app/pyproject.toml
# If you use a lock file (e.g., poetry.lock, pdm.lock, or requirements.txt derived from pyproject.toml),
# you should copy it here as well for reproducible builds:
# COPY ./poetry.lock /app/poetry.lock

# Install Python dependencies using uv.
# Dependencies are installed into the /install directory, which will be copied to the final stage.
# ".[postgresql]" installs dependencies from the current directory's pyproject.toml,
# including the optional 'postgresql' group.
#
# To leverage uv's own caching mechanism across multiple `docker build` runs (e.g., on a CI server or locally),
# you can use a cache mount. This speeds up the `uv pip install` step itself but doesn't affect the final image size.
# Example with cache mount:
# RUN --mount=type=cache,target=/root/.cache/uv \
#     uv pip install --prefix=/install ".[postgresql]"
RUN uv pip install --prefix=/install ".[postgresql]"
# Note: uv by default has behavior similar to --no-cache-dir for pip when not using a persistent UV_CACHE_DIR.
# The Docker layer caching (by copying pyproject.toml first) is the primary caching mechanism here for reproducibility.

# Stage 3: Final image
FROM base AS final

# Install runtime system dependencies.
# 'curl' is kept from your original Dockerfile; remove if not needed at runtime.
# If your application (e.g., psycopg2) requires libpq at runtime and it's not bundled
# with the Python package, you might need to install it (e.g., libpq5 or postgresql-client).
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy the installed Python packages from the builder stage's /install directory
# to the standard Python locations in the final image (/usr/local for python-slim).
COPY --from=builder /install /usr/local

# Copy your application code into the image
COPY . /app/

# Set the working directory for your application
WORKDIR /app/aqueduct

# Expose the port your application runs on
EXPOSE 8000

# Command to run your application using Daphne
CMD ["daphne", "-b", "0.0.0.0", "-p", "8000", "aqueduct.asgi:application"]

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY . /app/

# Install Python dependencies (including postgres dependencies)
RUN pip install --no-cache --upgrade pip \
    && pip install --no-cache ".[postgresql]"
WORKDIR /app/aqueduct

# Expose port for Daphne
EXPOSE 8000

# Use Daphne to serve the ASGI app
CMD ["daphne", "-b", "0.0.0.0", "-p", "8000", "aqueduct.asgi:application"]

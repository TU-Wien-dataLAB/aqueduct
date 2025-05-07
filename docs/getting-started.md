---
title: Getting Started
nav_order: 2
---

# Getting Started

Welcome! This guide will help you get up and running with the Aqueduct AI Gateway project.

## Quick Start (with Docker Compose)

The recommended way to get Aqueduct running locally is with Docker Compose. This will start the Django app, a PostgreSQL
database, and a local mock OIDC provider (Dex) for authentication.

1. **Clone the repository**
   ```bash
   git clone https://github.com/tu-wien-datalab/aqueduct.git
   cd aqueduct
   ```

2. **Start the services**
   ```bash
   docker compose up --build
   ```
   This will build and start all required services using the provided `.example.env` file for environment variables.

3. **Access the application**

    - The web UI will be available at [http://localhost:8000](http://localhost:8000)
    - The local OIDC provider (Dex) will be running at [http://localhost:5556/dex](http://localhost:5556/dex)
    - Default login credentials for Dex are:
        - **Username:** `you@example.com`
        - **Password:** `1234`

You can now access the admin UI and start exploring the gateway features.

---

## Local Setup (with `uv`)

If you prefer to run Aqueduct directly on your machine (for development), you can use [
`uv`](https://github.com/astral-sh/uv):

1. **Install `uv`**  
   If you don't have `uv` installed, you can install it via pip:
   ```bash
   pip install uv
   ```

2. **Clone the repository**
   ```bash
   git clone https://github.com/tu-wien-datalab/aqueduct.git
   cd aqueduct
   ```

3. **Create a virtual environment and install dependencies**
   ```bash
   uv venv
   source .venv/bin/activate
   uv sync
   ```

4. **Run the Django development server**
   ```bash
   uv run aqueduct/manage.py runserver
   ```

_Note: You will need to have Dex running locally, or adjust your environment variables accordingly._

---
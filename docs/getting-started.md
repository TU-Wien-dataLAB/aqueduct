---
title: Getting Started
nav_order: 2
---

# Getting Started
{: .no_toc }

## Table of contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

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

> [!NOTE]
> This starts Django in debug mode and is not suitable for production deployments. Change the [necessary settings](https://docs.djangoproject.com/en/5.2/topics/settings/#the-basics) for a production deployment.

### Starting vLLM (Optional)

To use an actual provider, you can run vLLM locally on your machine, e.g. with:

```bash
vllm serve Qwen/Qwen2.5-0.5B-Instruct -p 8001
```

For more information, follow the [Quickstart](https://docs.vllm.ai/en/stable/getting_started/quickstart.html) guide on vLLM.

Then follow the [User Guide](user-guide/models.md) to create an Endpoint with the (internal URL `http://host.docker.internal:8001/v1` and add the model with the name `Qwen/Qwen2.5-0.5B-Instruct` and a display name of your choosing.

You can then create a token and run the examples in the [User Guide](user-guide/examples.md), for example:

```bash
curl http://localhost:8000/vllm/models \
  -H "Authorization: Bearer YOUR_AQUEDUCT_TOKEN"
```

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

_Note: You will need to have Dex running locally, or adjust your environment variables accordingly to use an existing OIDC provider._

---

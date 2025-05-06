---
title: Getting Started
nav_order: 2
---

# Getting Started

Welcome! This guide will help you get up and running with the Aqueduct AI Gateway project.

## Quick Start

To set up the project for development using `uv`, follow these steps:

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

You can now access the admin UI and start exploring the gateway features.

---


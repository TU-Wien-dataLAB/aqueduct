<div align="center">
  <img src="./docs/assets/Aqueduct Icon.png" width="50%" alt="Aqueduct AI Gateway Logo" />
</div>

# Aqueduct AI Gateway

[![GitHub License](https://img.shields.io/github/license/tu-wien-datalab/aqueduct)](LICENSE)
[![GitHub last commit](https://img.shields.io/github/last-commit/tu-wien-datalab/aqueduct)](https://github.com/tu-wien-datalab/aqueduct/commits/main)
[![Project Status: WIP – Initial development is in progress, but there has not yet been a stable, usable release suitable for the public.](https://www.repostatus.org/badges/latest/wip.svg)](https://www.repostatus.org/#wip)
[![Static Badge](https://img.shields.io/badge/Documentation-GitHub%20Pages-brightgreen)](https://tu-wien-datalab.github.io/aqueduct/)


> [!NOTE]
> This project is still in active development! Contributions welcome!

**Aqueduct AI Gateway** aims to provide a **simple yet fully-featured** AI gateway you can self-host with **MCP server support**:

- no [SSO tax](https://konghq.com/pricing)
- no [observability tax](https://www.litellm.ai/enterprise)
- no [self-hosting tax](https://portkey.ai/pricing)
- no [org management tax](https://www.litellm.ai/enterprise)
- etc.

We aim to achieve this by:

- Building on top of the **LiteLLM Router SDK** to provide enhanced control and routing capabilities, while avoiding re-implementing the entire OpenAI-compatible APIs or every new feature, and
- Using **Django** for a clean, efficient, and maintainable implementation.

For more information, please read the [Documentation](https://tu-wien-datalab.github.io/aqueduct/).

If you don’t need user self-service, take a look at [Envoy AI Gateway](https://aigateway.envoyproxy.io) instead!

![AI Gateway Architecture](./docs/assets/screenshot.png "AI Gateway Architecture")

## Quick Start

The recommended way to get Aqueduct running locally is with Docker Compose. This will start the Django app, a PostgreSQL
database, Celery with Redis as the broker, Tika for text extraction from files, and a local mock OIDC provider (Dex)
for authentication.

1. **Clone the repository**
   ```bash
   git clone https://github.com/tu-wien-datalab/aqueduct.git
   cd aqueduct
   ```

2. **Set the necessary environment variables**
Most of the necessary environment variables are provided in the `.example.env` file.
You only need to set the `OPENAI_API_KEY`:
   ```bash
   export OPENAI_API_KEY="your-openai-api-key"
   ```
   This variable is used by the sample router configuration, provided in the
`example_router_config.yaml` file. Adjust it if you want to use other models.

3. **Start the services**
   ```bash
   docker compose up --build
   ```
   This will build and start all required services using the provided `.example.env` and
`example_router_config.yaml` files for environment variables and the router configuration.

4. **Access the application**

    - The web UI will be available at [http://localhost:8000](http://localhost:8000)
    - The local OIDC provider (Dex) will be running at [http://localhost:5556/dex](http://localhost:5556/dex)
    - Default login credentials for Dex are:
        - **Username:** `you@example.com`
        - **Password:** `1234`

You can now access the admin UI and start exploring the gateway features.

> **NOTE:**
> This starts Django in debug mode and is not suitable for production deployments. Change the
> [necessary settings](https://docs.djangoproject.com/en/5.2/topics/settings/#the-basics) for a production deployment.


For other installation methods, check out the [Getting Started Guide](https://tu-wien-datalab.github.io/aqueduct/getting-started/).

## Testing

```bash
cd aqueduct && python manage.py test
```

## Architecture

```mermaid
flowchart TB
    subgraph Client["Client Applications"]
        direction LR
        OpenAI[OpenAI SDK]
        HTTP[HTTP/REST Clients]
        MCP_Client[MCP Clients]
    end

    subgraph Gateway["Aqueduct AI Gateway (Django + ASGI)"]
        direction TB
        
        subgraph Auth["Authentication & Authorization"]
            TokenAuth[Token Authentication]
            OIDC[OIDC Integration]
            TOS[Terms of Service]
        end
        
        subgraph Router["Request Processing"]
            Limits[Rate Limiting]
            ModelCheck[Model Access Control]
            FileProc[File Processing]
            Logging[Request Logging]
        end
        
        subgraph API["API Endpoints"]
            ChatAPI[Chat Completions]
            CompAPI[Completions]
            EmbedAPI[Embeddings]
            BatchAPI[Batch Jobs]
            FileAPI[Files]
            SpeechAPI[Audio/Speech]
            ImageAPI[Image Generation]
            ModelsAPI[Models]
            MCPAPI[MCP Proxy]
        end
        
        LiteLLM[LiteLLM Router SDK]
    end
    
    subgraph Management["Management Layer"]
        direction TB
        AdminUI[Django Admin UI]
        
        subgraph Hierarchy["Organization Hierarchy"]
            Org[Organizations]
            Teams[Teams]
            Users[Users]
            Tokens[API Tokens]
        end
        
        subgraph Config["Configuration"]
            ModelExcl[Model Exclusions]
            RateLimits[Rate Limits]
            MCPConfig[MCP Server Config]
        end
    end
    
    subgraph Storage["Data & Storage"]
        PostgreSQL[(PostgreSQL)]
        Redis[(Redis Cache)]
        FileStore[File Storage]
    end
    
    subgraph Workers["Background Processing"]
        Celery[Celery Workers]
        Tika[Apache Tika<br/>Text Extraction]
    end
    
    subgraph External["External Services"]
        direction TB
        AIModels[AI Model Providers<br/>OpenAI, Anthropic, etc.]
        MCPServers[MCP Servers<br/>Tool Providers]
    end

    Client -->|API Requests| Auth
    Auth --> Router
    Router --> API
    API --> LiteLLM
    
    MCP_Client -.->|MCP Protocol| MCPAPI
    MCPAPI -.->|Proxy| MCPServers
    
    LiteLLM -->|Route Requests| AIModels
    
    AdminUI --> Hierarchy
    Hierarchy --> Config
    
    Auth -.->|Validate| PostgreSQL
    Router -.->|Check Limits| PostgreSQL
    Router -.->|Log Usage| PostgreSQL
    API -.->|Store Files| FileStore
    
    FileProc -->|Extract Text| Tika
    BatchAPI -->|Queue Jobs| Celery
    
    Celery -.->|Process| Redis
    Logging -.->|Cache| Redis
    
    Config -.->|Configure| LiteLLM
    
    classDef clientStyle fill:#e1f5ff,stroke:#01579b,stroke-width:2px
    classDef gatewayStyle fill:#fff3e0,stroke:#e65100,stroke-width:2px
    classDef mgmtStyle fill:#f3e5f5,stroke:#4a148c,stroke-width:2px
    classDef storageStyle fill:#e8f5e9,stroke:#1b5e20,stroke-width:2px
    classDef externalStyle fill:#fce4ec,stroke:#880e4f,stroke-width:2px
    
    class Client,OpenAI,HTTP,MCP_Client clientStyle
    class Gateway,Auth,Router,API,LiteLLM gatewayStyle
    class Management,AdminUI,Hierarchy,Config mgmtStyle
    class Storage,PostgreSQL,Redis,FileStore,Workers,Celery,Tika storageStyle
    class External,AIModels,MCPServers externalStyle
```

### Core Features
- **Token Authentication** - Django-based API token authentication with OIDC support
- **Model Access Control** - Hierarchical model exclusion lists (Org → Team → User)
- **Rate Limiting** - Multi-level limits (requests/min, input/output tokens per minute)
- **Usage Tracking** - Comprehensive request logging and analytics
- **File Processing** - Apache Tika integration for text extraction from documents

### Role-Based Access Control
- **User** - API key generation and personal usage
- **Team-Admin** - Team management and member access
- **Org-Admin** - Organization-wide user management
- **Admin** - Global settings and system configuration

### OpenAI-Compatible API
- Streaming and non-streaming completions
- Chat completions with file content support
- Embeddings generation
- Batch API for high-throughput workloads
- File upload and processing
- Audio speech synthesis and transcription
- Image generation

### MCP Integration
- MCP server management for enhanced tool calling
- Custom MCP server endpoints via HTTP proxy
- Session-based MCP protocol support
- DNS rebinding protection for security

## Links

[Documentation](https://tu-wien-datalab.github.io/aqueduct/) | [Issues](https://github.com/TU-Wien-dataLAB/aqueduct/issues)

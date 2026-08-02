# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Joysteed Agent Studio** is a production-grade multi-agent customer service platform. It orchestrates multiple specialized AI agents via LangGraph, backed by a hybrid RAG pipeline, 5 MCP tool servers, 3-layer memory, 3-tier stability engineering, and full-stack observability.

## Monorepo Structure

```
agent-studio/
├── agent-core/    # Python/LangGraph orchestration service (port 8000)
├── agent-rag/     # Python/FastAPI RAG knowledge base (port 8010)
├── agent-tools/   # Python MCP tool servers (ports 8001–8005)
├── agent-web/     # TypeScript/React customer chat UI (port 3001)
├── agent-admin/   # TypeScript/React RAG management console (port 3003)
├── agent-desk/    # TypeScript/React human operator workbench (port 3005)
├── docs/          # Architecture design docs
├── observability/ # Prometheus, Tempo, Grafana, OTEL configs
├── db-init/       # PostgreSQL initialization SQL
├── docker-compose.yml
├── litellm_config.yaml
└── Makefile
```

## Common Commands

### Full Stack
```bash
make install        # Install all Python (uv) + Node (npm) dependencies
make up             # docker compose up -d (all 20 services)
make down           # Stop all services
make build          # Rebuild Docker images
make test           # Run agent-core + agent-rag tests
make lint           # Run ruff on Python code
make clean          # Remove build artifacts, caches, volumes
```

### Python Services (agent-core, agent-rag, agent-tools)
```bash
# Setup (run inside each sub-project directory)
uv venv && uv pip install -e ".[dev]"   # agent-core / agent-tools
uv venv && uv pip install -e ".[vector,dev]"  # agent-rag (adds Milvus/fastembed)

# Run locally (agent-core)
uvicorn src.main:app --reload --port 8000

# Tests
pytest tests/ -v                        # all tests
pytest tests/test_agents/ -v            # single directory
pytest tests/test_agents/test_supervisor.py -v  # single file
pytest tests/ -k "test_retry"           # by keyword

# Linting
ruff check .
ruff check . --fix
```

### TypeScript Frontends (agent-web, agent-admin, agent-desk)
```bash
npm install
npm run dev       # Vite dev server
npm run build     # tsc + vite build
npm test          # vitest
npm run lint      # eslint
npm run format    # prettier
```

## Architecture

### Agent Orchestration (agent-core)

The core follows a **supervisor → business agent** pattern using LangGraph:

1. `agents/supervisor.py` classifies user intent and dispatches to one of 4 business agents (FAQ, Order, Complaint, Tech Support).
2. Each agent is defined in `config/agents.yaml` (agents-as-data) — adding a new agent requires only YAML + a prompt file, no code changes.
3. Agents call tools via MCP servers through `tools/executor.py`, which enforces per-tool authorization.
4. Sensitive operations (refunds, etc.) pass through `agents/approval.py` before execution.
5. After tool calls, `reflection/judge.py` can validate responses via a second LLM pass (configurable per agent via `reflection: judge | self_check | off`).

LangGraph state is checkpointed to PostgreSQL via `langgraph-checkpoint-postgres`.

### Memory System (agent-core/src/memory/)

3-layer memory managed by `memory/manager.py`:
- **L1 Working** (`working.py`) — in-request context, Redis-backed
- **L2 Short-term** (`short_term.py`) — conversation history in PostgreSQL + Redis cache
- **L3 Long-term** (`long_term/`) — vector embeddings in Milvus, with decay (`decay.py`) and entity extraction (`entities.py`)

Enable/disable with `MEMORY_ENABLED=true/false`.

### Stability / Guardrails (agent-core/src/guardrails/)

3-tier stability in `guardrails/engine.py`:
- **L1 Retry** (`retry.py`) — exponential backoff on transient failures
- **L2 Fallback** (`fallback.py`) — single-intent fallback on repeated failure
- **L3 Human handoff** (`agents/human_handoff.py`) — escalates to human operator

Enable/disable with `STABILITY_ENABLED=true/false`.

### RAG Pipeline (agent-rag)

Hybrid retrieval in `retrieval/milvus_backend.py`:
- **BM25** (sparse) + **vector search** (dense) fused via RRF (`vector_weight=0.6, bm25_weight=0.4`)
- Cross-encoder reranking (`rerank/`) applied on top candidates
- Embedding uses local fastembed ONNX models (no external API calls)
- Falls back to RRF if embedding or reranking fails

RAG settings are in `config.py` (`RAGSettings`) and controlled by env vars (`EMBEDDING_PROVIDER`, `RETRIEVAL_BACKEND`, `RERANK_PROVIDER`).

### MCP Tool Servers (agent-tools)

5 independent MCP servers, each with its own DB schema:
| Server | Port | Tools |
|--------|------|-------|
| knowledge_server | 8001 | search_faq, search_docs |
| order_server | 8002 | query_order, apply_refund, track_shipment |
| ticket_server | 8003 | create_ticket, query_ticket, update_ticket |
| crm_server | 8004 | get_customer_info, update_customer_tag |
| skill_server | 8005 | diagnose_fault (chains order + docs + ticket) |

Each server exposes an HTTP MCP endpoint. `agent-core` connects via `mcp_client/client.py`.

### Multi-Model Routing

`litellm_config.yaml` routes by task complexity via a LiteLLM proxy (port 4000):
- `model_fast` → Claude Haiku 4.5 (FAQ)
- `model_main` → Claude Sonnet 4.6 (Orders, Tech Support)
- `model_complex` → Claude Opus 4.8 (Complaints)
- Fallback: GPT-5.5 if Claude is unavailable

### Observability

All LLM calls are traced via Langfuse. OpenTelemetry spans flow to Tempo. Metrics go to Prometheus and are visualized in Grafana (dashboards: LLM token usage, agent latency, system overview).

## Key Configuration Files

| File | Purpose |
|------|---------|
| `agent-core/config/agents.yaml` | Agent registry — defines agents, prompts, tools, reflection mode |
| `agent-core/src/config.py` | All `Settings` classes (LLM, Redis, memory, stability, security) |
| `agent-rag/src/config.py` | `RAGSettings` (embedding provider, retrieval backend, rerank) |
| `litellm_config.yaml` | Multi-model routing, retry, fallback, Langfuse callback |
| `.env` / `.env.example` | API keys and runtime configuration |
| `docker-compose.yml` | All 20 services with env injection |

## Feature Flags (env vars)

| Variable | Default | Effect |
|----------|---------|--------|
| `STABILITY_ENABLED` | `true` | Enable guardrails (retry/fallback/handoff) |
| `MEMORY_ENABLED` | `true` | Enable 3-layer memory system |
| `REFLECTION_ENABLED` | `false` | Enable LLM self-reflection by default |
| `APPROVAL_ENABLED` | `true` | Require approval gate for sensitive ops |

## Test Patterns

Python tests use `pytest-asyncio` with `asyncio_mode = "auto"` — all async tests run without explicit `@pytest.mark.asyncio`.

`agent-rag` includes a `pseudo_embedder.py` for deterministic test embeddings (avoids loading ONNX models in CI).

Test fixtures are in each sub-project's `tests/conftest.py`.

## Database Layout

Three PostgreSQL databases on the same instance:
- `agent_core` — sessions, conversation history, LangGraph checkpoints, user auth, memory
- `agent_rag` — documents, chunks, embedding metadata, KB versioning
- `langfuse` — LLM traces, token usage, cost tracking

Initialized by `db-init/01-create-databases.sql` on first `docker compose up`.

FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

COPY pyproject.toml .
COPY shared/ shared/
COPY knowledge_server/ knowledge_server/
COPY order_server/ order_server/
COPY ticket_server/ ticket_server/
COPY crm_server/ crm_server/
COPY skill_server/ skill_server/

RUN uv pip install --system .

CMD ["python", "-m", "knowledge_server.server"]

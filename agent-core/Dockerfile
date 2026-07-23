FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

COPY pyproject.toml .
COPY src/ src/

RUN uv pip install --system .

# 运行时数据目录：agents.yaml 驱动 AGENT_REGISTRY、prompts 驱动各 Agent 提示词。
# 缺 config/ 会导致 registry 为空 → 路由到不存在的节点 → KeyError。
COPY config/ config/
COPY prompts/ prompts/

EXPOSE 8000

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]

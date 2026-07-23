FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# 依赖层：只拷依赖清单再装依赖。只要 pyproject.toml 不变，这一层就命中缓存，
# 改 src/ 下的业务代码不会触发依赖重装（原先 COPY src/ 在 install 之前，任何
# 代码改动都会让下面的 install 层失效 → 每次都重装全量依赖，构建慢的根因）。
COPY pyproject.toml .
RUN uv pip install --system -r pyproject.toml

# 源码/配置层：改这些只会让这之后的层失效，依赖层仍复用缓存。
# 运行时以 WORKDIR 下的 src/ 导入 src.main:app，无需把项目装成包。
COPY src/ src/
# 运行时数据目录：agents.yaml 驱动 AGENT_REGISTRY、prompts 驱动各 Agent 提示词。
# 缺 config/ 会导致 registry 为空 → 路由到不存在的节点 → KeyError。
COPY config/ config/
COPY prompts/ prompts/

EXPOSE 8000

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]

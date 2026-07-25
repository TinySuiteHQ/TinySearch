FROM python:3.12-slim AS runtime-base

ARG TINYSEARCH_VERSION=dev

LABEL org.opencontainers.image.title="TinySearch" \
      org.opencontainers.image.description="Local-first web research API with hybrid retrieval and ONNX embeddings" \
      org.opencontainers.image.licenses="MIT" \
      org.opencontainers.image.source="https://github.com/MarcellM01/TinySearch" \
      org.opencontainers.image.version="${TINYSEARCH_VERSION}"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TINYSEARCH_VERSION=${TINYSEARCH_VERSION} \
    TINYSEARCH_MODELS_DIR=/data/models \
    PLAYWRIGHT_BROWSERS_PATH=/ms-playwright

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl gosu \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# Browsers in a world-readable path so the non-root runtime user can launch Chromium.
RUN mkdir -p /ms-playwright \
    && pip install --upgrade pip \
    && pip install -r requirements.txt \
    && python -m playwright install --with-deps chromium \
    && chmod -R a+rX /ms-playwright

COPY . .

RUN useradd --create-home --shell /usr/sbin/nologin tinysearch \
    && mkdir -p /data/models /app/trace_logs \
    && chown -R tinysearch:tinysearch /data /app/trace_logs \
    && chmod +x /app/docker-entrypoint.sh

ENTRYPOINT ["/app/docker-entrypoint.sh"]

FROM runtime-base AS fastapi

LABEL org.opencontainers.image.title="TinySearch FastAPI" \
      org.opencontainers.image.description="TinySearch HTTP API server"

EXPOSE 8000
VOLUME ["/data/models"]
CMD ["uvicorn", "servers.fastapi_server:app", "--host", "0.0.0.0", "--port", "8000"]

FROM runtime-base AS mcp

LABEL org.opencontainers.image.title="TinySearch MCP" \
      org.opencontainers.image.description="TinySearch MCP stdio server" \
      io.modelcontextprotocol.server.name="io.github.MarcellM01/tinysearch"

VOLUME ["/data/models"]
CMD ["python", "servers/mcp_server.py"]

FROM python:3.12-slim

ARG TINYSEARCH_VERSION=dev

LABEL org.opencontainers.image.title="TinySearch" \
      org.opencontainers.image.description="TinySearch MCP and HTTP research server" \
      org.opencontainers.image.licenses="MIT" \
      org.opencontainers.image.source="https://github.com/TinySuiteHQ/TinySearch" \
      org.opencontainers.image.version="${TINYSEARCH_VERSION}" \
      io.modelcontextprotocol.server.name="io.github.TinySuiteHQ/tinysearch"

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

COPY . .

RUN mkdir -p /ms-playwright \
    && pip install --upgrade pip \
    && pip install ".[server]" \
    && python -m playwright install --with-deps chromium \
    && chmod -R a+rX /ms-playwright \
    && useradd --create-home --shell /usr/sbin/nologin tinysearch \
    && mkdir -p /data/models /app/trace_logs \
    && chown -R tinysearch:tinysearch /data /app/trace_logs \
    && chmod +x /app/docker-entrypoint.sh

EXPOSE 8000
VOLUME ["/data/models"]
HEALTHCHECK --interval=30s --timeout=15s --start-period=90s --retries=3 \
    CMD tinysearch doctor || exit 1
ENTRYPOINT ["/app/docker-entrypoint.sh"]
CMD ["tinysearch", "mcp"]

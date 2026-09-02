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
    PLAYWRIGHT_BROWSERS_PATH=/ms-playwright \
    NPM_CONFIG_CACHE=/home/tinysearch/.npm

WORKDIR /app

RUN apt-get update \
    && apt-get upgrade -y --no-install-recommends \
    && apt-get install -y --no-install-recommends ca-certificates curl gosu gnupg \
    && curl -fsSL https://deb.nodesource.com/setup_22.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && rm -rf /var/lib/apt/lists/*

COPY . .

RUN mkdir -p /ms-playwright \
    && pip install --upgrade pip \
    && pip install ".[server,telemetry]" \
    && python -m playwright install --with-deps chromium \
    && pip check \
    && pip uninstall --yes pip setuptools \
    && chmod -R a+rX /ms-playwright \
    && useradd --create-home --shell /usr/sbin/nologin tinysearch \
    && mkdir -p /data/models /app/trace_logs /home/tinysearch/.npm \
    && chown -R tinysearch:tinysearch /data /app/trace_logs /home/tinysearch/.npm \
    && chmod +x /app/docker-entrypoint.sh

EXPOSE 8000
VOLUME ["/data/models"]
HEALTHCHECK --interval=30s --timeout=15s --start-period=90s --retries=3 \
    CMD tinysearch doctor || exit 1
ENTRYPOINT ["/app/docker-entrypoint.sh"]
CMD ["tinysearch", "mcp"]

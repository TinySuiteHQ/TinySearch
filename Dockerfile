# syntax=docker/dockerfile:1

# uv installs exactly the versions pinned in uv.lock. Pinned by digest so a
# rebuild can't pick up a different uv. It is mounted for the steps that use it,
# not copied in: the running image never needs it.
FROM ghcr.io/astral-sh/uv:0.12.22@sha256:f513a91fc62fe7c17567eee97230dd198e43edb8a9fbecca843714a4358fe1bc AS uv

FROM python:3.12-slim

ARG TINYSEARCH_VERSION=dev

LABEL org.opencontainers.image.title="TinySearch" \
      org.opencontainers.image.description="TinySearch MCP and HTTP research server" \
      org.opencontainers.image.licenses="MIT" \
      org.opencontainers.image.source="https://github.com/TinySuiteHQ/TinySearch" \
      org.opencontainers.image.version="${TINYSEARCH_VERSION}" \
      io.modelcontextprotocol.server.name="io.github.TinySuiteHQ/tinysearch"

# UV_COMPILE_BYTECODE writes .pyc files at build time, so the server doesn't
# compile every module on each start (PYTHONDONTWRITEBYTECODE only stops writes
# at runtime). The project environment lives at /app/.venv, first on PATH.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    PATH="/app/.venv/bin:${PATH}" \
    TINYSEARCH_VERSION=${TINYSEARCH_VERSION} \
    TINYSEARCH_MODELS_DIR=/data/models \
    PLAYWRIGHT_BROWSERS_PATH=/ms-playwright

WORKDIR /app

RUN apt-get update \
    && apt-get upgrade -y --no-install-recommends \
    && apt-get install -y --no-install-recommends ca-certificates gosu libatomic1 \
    && rm -rf /var/lib/apt/lists/*

# Dependencies and Chromium depend only on the lockfile, so these layers are
# reused until uv.lock changes; a source-only change skips straight to the
# project install below.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv --mount=from=uv,source=/uv,target=/bin/uv \
    uv sync --locked --no-dev --no-install-project --extra server --extra telemetry

# Only Chrome Headless Shell: TinySearch launches Chromium headless, which
# Playwright never serves from full Chromium. Its system libraries are listed
# here rather than taken from --with-deps, which installs what a windowed
# Chromium needs (a virtual display, a software GPU stack, many fonts). One
# font family is enough to lay pages out; text is read from the page, not drawn.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       libasound2t64 libatk1.0-0t64 libatspi2.0-0t64 libdbus-1-3 libexpat1 libgbm1 \
       libglib2.0-0t64 libnspr4 libnss3 libx11-6 libxcb1 libxcomposite1 libxdamage1 \
       libxext6 libxfixes3 libxkbcommon0 libxrandr2 fonts-liberation \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /ms-playwright \
    && playwright install --only-shell chromium \
    && chmod -R a+rX /ms-playwright

COPY . .
RUN --mount=type=cache,target=/root/.cache/uv --mount=from=uv,source=/uv,target=/bin/uv \
    uv sync --locked --no-dev --no-editable --extra server --extra telemetry \
    && uv pip check --python /app/.venv/bin/python \
    && /usr/local/bin/python3 -m pip uninstall --yes pip setuptools \
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

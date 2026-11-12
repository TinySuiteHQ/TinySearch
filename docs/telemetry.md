# OpenTelemetry export

TinySearch emits vendor-neutral traces and metrics only when OpenTelemetry is
explicitly configured. Normal library, MCP, and FastAPI behavior is unchanged
when telemetry is not installed or not configured.

Install the optional exporter support for a standalone MCP server:

```bash
uvx --from "tinysuite-search[server,telemetry]" tinysearch
```

The official Docker image includes the same optional support. Set standard OTel
variables on either deployment; the common endpoint enables traces and metrics:

```bash
OTEL_SERVICE_NAME=tinysearch \
OTEL_EXPORTER_OTLP_ENDPOINT=http://collector:4318 \
OTEL_EXPORTER_OTLP_PROTOCOL=http/protobuf \
tinysearch serve
```

`http/protobuf` is the default and `grpc` is also supported. Use
`OTEL_TRACES_EXPORTER=none`, `OTEL_METRICS_EXPORTER=none`, or
`OTEL_SDK_DISABLED=true` to disable telemetry. Standard resource, header,
timeout, sampler, and signal-specific endpoint settings are passed through to
the OpenTelemetry SDK; treat `OTEL_EXPORTER_OTLP_HEADERS` as a secret.

## What is exported

TinySearch exports operation/stage timing, outcomes, counts, backend state,
browser use, token counts, and embedding model metadata. It never exports
queries, URLs, domains, prompts, documents, snippets, request headers,
credentials, configuration paths, raw errors, exception stacks, MCP arguments,
or MCP results.

Direct Python-library users configure their own OTel provider; TinySearch only
auto-configures its standalone MCP and FastAPI server entry points.

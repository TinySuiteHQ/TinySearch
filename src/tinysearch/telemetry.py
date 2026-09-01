"""Optional, transport-neutral OpenTelemetry support for TinySearch.

The public API dependency is deliberately lightweight: without an SDK provider,
all spans and metrics emitted here are no-ops.  Standalone servers can opt in
to the SDK/exporters through :func:`configure_from_environment`; library users
keep ownership of their application's provider setup.
"""

from __future__ import annotations

import contextlib
import importlib.metadata
import os
import sys
import time
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any

from opentelemetry import metrics, trace
from opentelemetry.trace import Status, StatusCode


_SCOPE_NAME = "tinysearch"
_WARNED: set[str] = set()
_OWNED_PROVIDERS: list[Any] = []
_CONFIGURED = False


def _version() -> str:
    try:
        return importlib.metadata.version("tinysuite-search")
    except importlib.metadata.PackageNotFoundError:
        return "dev"


def _warn_once(code: str, message: str) -> None:
    if code in _WARNED:
        return
    _WARNED.add(code)
    print(f"[telemetry] {message}", file=sys.stderr, flush=True)


def _safe_attributes(attributes: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Keep instrumentation values scalar and controlled by TinySearch code.

    Call sites must pass only low-cardinality, non-sensitive operational values.
    This helper intentionally does not inspect user input or configuration.
    """

    if not attributes:
        return {}
    return {
        str(key): value
        for key, value in attributes.items()
        if isinstance(value, (str, bool, int, float)) and value is not None
    }


def _outcome_is_error(outcome: str) -> bool:
    return outcome in {"error", "partial", "timeout", "search_backend_error"}


@dataclass
class SpanScope:
    """A span that records a sanitized status when it completes."""

    span: Any
    started_at: float
    operation: str | None = None
    record_operation_metric: bool = False
    _outcome: str = "ok"
    _result_count: int = 0
    _error_type: str | None = None
    _completed: bool = False
    _extra_attributes: dict[str, Any] = field(default_factory=dict)

    def set_attributes(self, attributes: Mapping[str, Any]) -> None:
        safe = _safe_attributes(attributes)
        self._extra_attributes.update(safe)
        for key, value in safe.items():
            self.span.set_attribute(key, value)

    def complete(
        self,
        *,
        outcome: str = "ok",
        result_count: int = 0,
        error_type: str | None = None,
        attributes: Mapping[str, Any] | None = None,
    ) -> None:
        if self._completed:
            return
        self._completed = True
        self._outcome = outcome
        self._result_count = max(0, int(result_count))
        self._error_type = error_type
        if attributes:
            self.set_attributes(attributes)
        self.span.set_attribute("tinysearch.operation.status", outcome)
        if error_type:
            self.span.set_attribute("error.type", error_type)
        if _outcome_is_error(outcome) or error_type:
            # Deliberately do not call record_exception(): exception messages and
            # stacks can contain query text, URLs, credentials, or page content.
            self.span.set_status(Status(StatusCode.ERROR))

    def fail(self, exc: BaseException) -> None:
        self.complete(outcome="error", error_type=type(exc).__name__)

    def finish(self) -> None:
        if not self._completed:
            self.complete()
        if self.record_operation_metric and self.operation:
            elapsed = max(0.0, time.perf_counter() - self.started_at)
            attributes = {
                "tinysearch.operation.name": self.operation,
                "tinysearch.operation.status": self._outcome,
            }
            if self._error_type:
                attributes["error.type"] = self._error_type
            _operation_duration().record(elapsed, attributes)
            _operation_result_count().record(self._result_count, attributes)


@contextlib.contextmanager
def span_scope(
    name: str,
    *,
    attributes: Mapping[str, Any] | None = None,
    operation: str | None = None,
    record_operation_metric: bool = False,
) -> Iterator[SpanScope]:
    """Create a current span without exposing telemetry implementation details."""

    safe_attributes = _safe_attributes(attributes)
    if operation:
        safe_attributes.setdefault("tinysearch.operation.name", operation)
    tracer = trace.get_tracer(_SCOPE_NAME, _version())
    with tracer.start_as_current_span(name, attributes=safe_attributes) as span:
        scope = SpanScope(
            span=span,
            started_at=time.perf_counter(),
            operation=operation,
            record_operation_metric=record_operation_metric,
        )
        try:
            yield scope
        except BaseException as exc:
            scope.fail(exc)
            raise
        finally:
            scope.finish()


@contextlib.contextmanager
def backend_attempt_scope(
    *,
    backend: str,
    attempt: int,
    fallback: bool,
) -> Iterator[SpanScope]:
    """Record one logical web-search backend attempt and its duration metric."""

    started_at = time.perf_counter()
    with span_scope(
        "tinysearch.search.backend",
        attributes={
            "tinysearch.search.backend": backend,
            "tinysearch.search.attempt": attempt,
            "tinysearch.search.fallback": fallback,
        },
        operation="search_backend",
    ) as scope:
        try:
            yield scope
        finally:
            if not scope._completed:
                scope.complete()
            attributes = {
                "tinysearch.search.backend": backend,
                "tinysearch.search.state": scope._outcome,
                "tinysearch.search.fallback": fallback,
            }
            if scope._error_type:
                attributes["error.type"] = scope._error_type
            _backend_duration().record(max(0.0, time.perf_counter() - started_at), attributes)


def _operation_duration() -> Any:
    return metrics.get_meter(_SCOPE_NAME, _version()).create_histogram(
        "tinysearch.operation.duration",
        unit="s",
        description="Duration of TinySearch operations.",
    )


def _operation_result_count() -> Any:
    return metrics.get_meter(_SCOPE_NAME, _version()).create_histogram(
        "tinysearch.operation.result.count",
        unit="{result}",
        description="Number of TinySearch operation outputs.",
    )


def _backend_duration() -> Any:
    return metrics.get_meter(_SCOPE_NAME, _version()).create_histogram(
        "tinysearch.search.backend.duration",
        unit="s",
        description="Duration of TinySearch web-search backend attempts.",
    )


def _env_true(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _signal_enabled(signal: str) -> bool:
    exporter = os.environ.get(f"OTEL_{signal.upper()}_EXPORTER", "").strip().lower()
    endpoint_present = bool(
        os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", "").strip()
        or os.environ.get(f"OTEL_EXPORTER_OTLP_{signal.upper()}_ENDPOINT", "").strip()
    )
    if exporter:
        values = {part.strip() for part in exporter.split(",") if part.strip()}
        if "none" in values:
            return False
        if "otlp" in values:
            return True
        if endpoint_present:
            _warn_once(
                f"unsupported-{signal}-exporter",
                f"OTEL_{signal.upper()}_EXPORTER does not select OTLP; TinySearch will not configure {signal} export.",
            )
        return False
    return endpoint_present


def _protocol_for(signal: str) -> str | None:
    protocol = (
        os.environ.get(f"OTEL_EXPORTER_OTLP_{signal.upper()}_PROTOCOL", "").strip().lower()
        or os.environ.get("OTEL_EXPORTER_OTLP_PROTOCOL", "").strip().lower()
        or "http/protobuf"
    )
    if protocol in {"http/protobuf", "grpc"}:
        return protocol
    _warn_once(
        f"unsupported-{signal}-protocol-{protocol}",
        f"OTLP {signal} protocol is unsupported; use http/protobuf or grpc. {signal.capitalize()} export is disabled.",
    )
    return None


def _resource() -> Any:
    from opentelemetry.sdk.resources import Resource

    resource = Resource.create()
    attributes = dict(resource.attributes)
    # Resource.create() applies OTEL_RESOURCE_ATTRIBUTES and OTEL_SERVICE_NAME.
    # Only supply TinySearch defaults when the operator has not supplied values.
    if not attributes.get("service.name") or attributes.get("service.name") == "unknown_service":
        attributes["service.name"] = "tinysearch"
    attributes.setdefault("service.version", _version())
    return Resource(attributes=attributes)


def _has_application_provider(signal: str) -> bool:
    provider = (
        trace.get_tracer_provider()
        if signal == "traces"
        else metrics.get_meter_provider()
    )
    return type(provider).__name__ not in {"ProxyTracerProvider", "_ProxyMeterProvider"}


def _build_trace_provider(protocol: str) -> Any:
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    if protocol == "grpc":
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    else:
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    provider = TracerProvider(resource=_resource())
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
    return provider


def _build_meter_provider(protocol: str) -> Any:
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader

    if protocol == "grpc":
        from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
    else:
        from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
    return MeterProvider(
        resource=_resource(),
        metric_readers=[PeriodicExportingMetricReader(OTLPMetricExporter())],
    )


def configure_from_environment() -> None:
    """Configure OTLP export for a standalone TinySearch server when opted in.

    No TinySearch-specific environment variable is required.  Import and setup
    failures are deliberately isolated so an observability deployment cannot
    prevent the server from handling requests.
    """

    global _CONFIGURED
    if _CONFIGURED or _env_true("OTEL_SDK_DISABLED"):
        return
    _CONFIGURED = True
    if _signal_enabled("traces") and not _has_application_provider("traces"):
        protocol = _protocol_for("traces")
        if protocol:
            try:
                provider = _build_trace_provider(protocol)
                trace.set_tracer_provider(provider)
                _OWNED_PROVIDERS.append(provider)
            except Exception:  # noqa: BLE001 - telemetry setup must be non-blocking
                _warn_once(
                    "trace-setup-failed",
                    "could not configure OTLP trace export; TinySearch will continue without it.",
                )
    if _signal_enabled("metrics") and not _has_application_provider("metrics"):
        protocol = _protocol_for("metrics")
        if protocol:
            try:
                provider = _build_meter_provider(protocol)
                metrics.set_meter_provider(provider)
                _OWNED_PROVIDERS.append(provider)
            except Exception:  # noqa: BLE001 - telemetry setup must be non-blocking
                _warn_once(
                    "metric-setup-failed",
                    "could not configure OTLP metric export; TinySearch will continue without it.",
                )


def shutdown() -> None:
    """Flush and stop only providers created by :func:`configure_from_environment`."""

    while _OWNED_PROVIDERS:
        provider = _OWNED_PROVIDERS.pop()
        try:
            provider.force_flush()
            provider.shutdown()
        except Exception:  # noqa: BLE001 - shutdown must not mask server shutdown
            _warn_once(
                "shutdown-failed",
                "could not flush telemetry during shutdown.",
            )

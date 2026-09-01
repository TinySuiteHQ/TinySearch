from __future__ import annotations

import asyncio
import os
import subprocess
import sys
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

from opentelemetry import metrics, trace
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode

from tinysearch import core
from tinysearch.services.web_search_service import SearchResult


_ROOT = Path(__file__).resolve().parents[1]


def _metric_names(reader: InMemoryMetricReader) -> set[str]:
    data = reader.get_metrics_data()
    if data is None:
        return set()
    return {
        metric.name
        for resource_metric in data.resource_metrics
        for scope_metric in resource_metric.scope_metrics
        for metric in scope_metric.metrics
    }


class CoreTelemetryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.exporter = InMemorySpanExporter()
        cls.tracer_provider = TracerProvider()
        cls.tracer_provider.add_span_processor(SimpleSpanProcessor(cls.exporter))
        trace.set_tracer_provider(cls.tracer_provider)

        cls.metric_reader = InMemoryMetricReader()
        cls.meter_provider = MeterProvider(metric_readers=[cls.metric_reader])
        metrics.set_meter_provider(cls.meter_provider)

    def setUp(self) -> None:
        self.exporter.clear()

    def test_search_spans_metrics_and_privacy(self) -> None:
        result = SearchResult(
            result_id=1,
            title="Sensitive title",
            url="https://private.example/path?token=secret",
            text="Sensitive retrieved content",
        )
        with patch(
            "tinysearch.services.web_search_service._backend_attempt_plan",
            return_value=[("ddgs", lambda: [result])],
        ):
            payload = asyncio.run(
                core.search(
                    [{"query": "secret user question", "domains": ["private.example"]}],
                    config={"search_backend": "ddgs"},
                )
            )

        self.assertEqual(payload["status"], "ok")
        spans = self.exporter.get_finished_spans()
        names = {span.name for span in spans}
        self.assertTrue(
            {"tinysearch.search", "tinysearch.search.item", "tinysearch.search.backend"}
            .issubset(names)
        )
        parent = next(span for span in spans if span.name == "tinysearch.search")
        item = next(span for span in spans if span.name == "tinysearch.search.item")
        backend = next(span for span in spans if span.name == "tinysearch.search.backend")
        self.assertEqual(item.parent.span_id, parent.context.span_id)
        self.assertEqual(backend.parent.span_id, item.context.span_id)
        self.assertEqual(parent.attributes["tinysearch.result.count"], 1)

        exported = "\n".join(
            str(value)
            for span in spans
            for value in (*span.attributes.values(), *(event.attributes for event in span.events))
        )
        for forbidden in (
            "secret user question",
            "private.example",
            "token=secret",
            "Sensitive title",
            "Sensitive retrieved content",
        ):
            self.assertNotIn(forbidden, exported)
        self.assertTrue(
            {
                "tinysearch.operation.duration",
                "tinysearch.operation.result.count",
                "tinysearch.search.backend.duration",
            }.issubset(_metric_names(self.metric_reader))
        )

    def test_partial_search_uses_standard_error_status_without_exception_event(self) -> None:
        payload = asyncio.run(core.search([{"query": ""}]))

        self.assertEqual(payload["status"], "partial")
        parent = next(
            span for span in self.exporter.get_finished_spans() if span.name == "tinysearch.search"
        )
        self.assertEqual(parent.status.status_code, StatusCode.ERROR)
        self.assertEqual(parent.attributes["error.type"], "partial_failure")
        self.assertEqual(parent.events, ())


class _Receiver(BaseHTTPRequestHandler):
    paths: list[str] = []

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        length = int(self.headers.get("Content-Length", "0"))
        self.rfile.read(length)
        type(self).paths.append(self.path)
        self.send_response(200)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def log_message(self, _format: str, *_args: object) -> None:
        return


class BootstrapTelemetryTests(unittest.TestCase):
    def _run_isolated(self, code: str, extra_env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
        environment = os.environ.copy()
        environment.pop("OTEL_EXPORTER_OTLP_ENDPOINT", None)
        environment.pop("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", None)
        environment.pop("OTEL_EXPORTER_OTLP_METRICS_ENDPOINT", None)
        environment.pop("OTEL_TRACES_EXPORTER", None)
        environment.pop("OTEL_METRICS_EXPORTER", None)
        environment.pop("OTEL_EXPORTER_OTLP_PROTOCOL", None)
        environment.pop("OTEL_EXPORTER_OTLP_TRACES_PROTOCOL", None)
        environment.pop("OTEL_EXPORTER_OTLP_METRICS_PROTOCOL", None)
        environment.pop("OTEL_SDK_DISABLED", None)
        environment["PYTHONPATH"] = str(_ROOT / "src")
        if extra_env:
            environment.update(extra_env)
        return subprocess.run(
            [sys.executable, "-c", code],
            cwd=_ROOT,
            env=environment,
            text=True,
            capture_output=True,
            check=False,
        )

    def test_no_configuration_keeps_the_default_provider(self) -> None:
        process = self._run_isolated(
            "from tinysearch.telemetry import configure_from_environment; "
            "from opentelemetry import trace; configure_from_environment(); "
            "print(type(trace.get_tracer_provider()).__name__)"
        )

        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertEqual(process.stdout.strip(), "ProxyTracerProvider")

    def test_sdk_disabled_wins_over_an_otlp_endpoint(self) -> None:
        process = self._run_isolated(
            "from tinysearch.telemetry import configure_from_environment; "
            "from opentelemetry import trace; configure_from_environment(); "
            "print(type(trace.get_tracer_provider()).__name__)",
            {
                "OTEL_EXPORTER_OTLP_ENDPOINT": "http://127.0.0.1:4318",
                "OTEL_SDK_DISABLED": "true",
            },
        )

        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertEqual(process.stdout.strip(), "ProxyTracerProvider")

    def test_signal_none_overrides_a_common_endpoint(self) -> None:
        process = self._run_isolated(
            "from tinysearch.telemetry import configure_from_environment; "
            "from opentelemetry import metrics, trace; configure_from_environment(); "
            "print(type(trace.get_tracer_provider()).__name__, type(metrics.get_meter_provider()).__name__)",
            {
                "OTEL_EXPORTER_OTLP_ENDPOINT": "http://127.0.0.1:4318",
                "OTEL_TRACES_EXPORTER": "none",
            },
        )

        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertEqual(process.stdout.strip(), "ProxyTracerProvider MeterProvider")

    def test_existing_library_provider_is_preserved(self) -> None:
        process = self._run_isolated(
            "from opentelemetry import trace; from opentelemetry.sdk.trace import TracerProvider; "
            "from tinysearch.telemetry import configure_from_environment; p=TracerProvider(); "
            "trace.set_tracer_provider(p); configure_from_environment(); "
            "print(trace.get_tracer_provider() is p)",
            {"OTEL_EXPORTER_OTLP_ENDPOINT": "http://127.0.0.1:4318"},
        )

        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertEqual(process.stdout.strip(), "True")

    def test_invalid_protocol_disables_the_signal_without_leaking_endpoint(self) -> None:
        endpoint = "http://private-collector.invalid:4318"
        process = self._run_isolated(
            "from tinysearch.telemetry import configure_from_environment; "
            "from opentelemetry import trace; configure_from_environment(); "
            "print(type(trace.get_tracer_provider()).__name__)",
            {
                "OTEL_EXPORTER_OTLP_ENDPOINT": endpoint,
                "OTEL_EXPORTER_OTLP_PROTOCOL": "unsupported",
                "OTEL_METRICS_EXPORTER": "none",
            },
        )

        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertEqual(process.stdout.strip(), "ProxyTracerProvider")
        self.assertNotIn(endpoint, process.stderr)

    def test_common_endpoint_exports_traces_and_metrics_over_http(self) -> None:
        _Receiver.paths = []
        receiver = ThreadingHTTPServer(("127.0.0.1", 0), _Receiver)
        thread = threading.Thread(target=receiver.serve_forever, daemon=True)
        thread.start()
        try:
            endpoint = f"http://127.0.0.1:{receiver.server_port}"
            process = self._run_isolated(
                "from tinysearch.telemetry import configure_from_environment, shutdown, span_scope\n"
                "configure_from_environment()\n"
                "with span_scope('tinysearch.test', operation='test', record_operation_metric=True) as scope:\n"
                "    scope.complete(result_count=1)\n"
                "shutdown()\n"
                "print('ok')\n",
                {"OTEL_EXPORTER_OTLP_ENDPOINT": endpoint},
            )
        finally:
            receiver.shutdown()
            receiver.server_close()
            thread.join(timeout=5)

        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertEqual(process.stdout.strip(), "ok")
        self.assertIn("/v1/traces", _Receiver.paths)
        self.assertIn("/v1/metrics", _Receiver.paths)


class ServerBootstrapTests(unittest.TestCase):
    def test_fastapi_lifespan_configures_and_shuts_down_telemetry(self) -> None:
        from tinysearch.servers import fastapi_server

        async def exercise() -> None:
            async with fastapi_server._lifespan(fastapi_server.app):
                pass

        with (
            patch("tinysearch.servers.fastapi_server.configure_from_environment") as configure,
            patch("tinysearch.servers.fastapi_server.shutdown_telemetry") as shutdown,
        ):
            asyncio.run(exercise())

        configure.assert_called_once_with()
        shutdown.assert_called_once_with()

    def test_mcp_entrypoint_configures_and_shuts_down_telemetry(self) -> None:
        from tinysearch.servers import mcp_server

        with (
            patch.object(mcp_server, "_enable_traceback_dump"),
            patch.object(mcp_server, "configure_from_environment") as configure,
            patch.object(mcp_server, "shutdown_telemetry") as shutdown,
            patch.object(mcp_server.mcp, "run") as run,
            patch.dict(os.environ, {"MCP_TRANSPORT": "stdio"}),
        ):
            mcp_server.main()

        configure.assert_called_once_with()
        run.assert_called_once_with(transport="stdio")
        shutdown.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()

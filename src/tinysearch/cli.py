"""The `tinysearch` console command.

- No subcommand: run the stdio MCP server (the default `uvx`-spawned path).
- `setup [--with-system-deps]`: install Chromium and download the model.
- `doctor`: check readiness without downloading anything.
- `serve`: run the Streamable HTTP server.
"""

from __future__ import annotations

import argparse
import os
import sys


def _server_dependency_error(exc: ModuleNotFoundError) -> int:
    print(
        "TinySearch server dependencies are not installed.\n"
        'Install them with: pip install "tinysuite-tinysearch[server]"',
        file=sys.stderr,
    )
    return 2


def _run_mcp_stdio() -> int:
    try:
        from tinysearch.servers.mcp_server import main as mcp_main
    except ModuleNotFoundError as exc:
        return _server_dependency_error(exc)

    mcp_main()
    return 0


def _run_serve() -> int:
    os.environ.setdefault("MCP_TRANSPORT", "streamable-http")
    try:
        from tinysearch.servers.mcp_server import main as mcp_main
    except ModuleNotFoundError as exc:
        return _server_dependency_error(exc)

    mcp_main()
    return 0


def _run_setup(with_system_deps: bool) -> int:
    from tinysearch.setup import run as setup_run

    return setup_run(with_system_deps=with_system_deps)


def _run_doctor() -> int:
    from tinysearch.doctor import run as doctor_run

    return doctor_run()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tinysearch", description="TinySearch MCP server and tools."
    )
    subparsers = parser.add_subparsers(dest="command")

    setup_parser = subparsers.add_parser(
        "setup", help="Install Chromium and download the configured ONNX model."
    )
    setup_parser.add_argument(
        "--with-system-deps",
        action="store_true",
        help="Also install Chromium's OS-level dependencies (Linux only).",
    )

    subparsers.add_parser(
        "doctor",
        help="Check configuration, browser, model, and directory readiness "
        "without downloading anything.",
    )
    subparsers.add_parser("mcp", help="Run the stdio MCP server.")
    subparsers.add_parser("serve", help="Run the Streamable HTTP server.")

    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    if args.command == "setup":
        sys.exit(_run_setup(args.with_system_deps))
    elif args.command == "doctor":
        sys.exit(_run_doctor())
    elif args.command == "serve":
        sys.exit(_run_serve())
    elif args.command == "mcp":
        sys.exit(_run_mcp_stdio())
    else:
        sys.exit(_run_mcp_stdio())


if __name__ == "__main__":
    main()

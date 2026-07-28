"""Fail when Python, MCP Registry, and container release metadata drift."""

from __future__ import annotations

import argparse
import json
import re
import sys
import tomllib
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
GITHUB_REPOSITORY = "TinySuiteHQ/TinySearch"
GITHUB_URL = f"https://github.com/{GITHUB_REPOSITORY}"
SERVER_NAME = "io.github.TinySuiteHQ/tinysearch"
PYPI_PACKAGE = "tinysuite-search"
OCI_REPOSITORY = "docker.io/marcellm01/tinysearch"


def _one_package(server: dict[str, Any], registry_type: str) -> dict[str, Any]:
    packages = [
        package
        for package in server.get("packages", [])
        if package.get("registryType") == registry_type
    ]
    if len(packages) != 1:
        raise ValueError(
            f"server.json must contain exactly one {registry_type!r} package; "
            f"found {len(packages)}"
        )
    return packages[0]


def check(expected_version: str | None = None) -> str:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    server = json.loads((ROOT / "server.json").read_text(encoding="utf-8"))
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")

    project = pyproject["project"]
    version = str(project["version"])
    if expected_version is not None and version != expected_version.removeprefix("v"):
        raise ValueError(
            f"release version {expected_version!r} does not match pyproject.toml "
            f"version {version!r}"
        )
    if server.get("version") != version:
        raise ValueError(
            f"server.json version {server.get('version')!r} does not match "
            f"pyproject.toml version {version!r}"
        )
    if server.get("name") != SERVER_NAME:
        raise ValueError(f"server.json name must be {SERVER_NAME!r}")
    if server.get("websiteUrl") != GITHUB_URL:
        raise ValueError(f"server.json websiteUrl must be {GITHUB_URL!r}")
    if server.get("repository", {}).get("url") != GITHUB_URL:
        raise ValueError(f"server.json repository.url must be {GITHUB_URL!r}")

    project_urls = project.get("urls", {})
    for label in ("Homepage", "Repository"):
        if project_urls.get(label) != GITHUB_URL:
            raise ValueError(f"project.urls.{label} must be {GITHUB_URL!r}")

    pypi = _one_package(server, "pypi")
    if pypi.get("identifier") != PYPI_PACKAGE:
        raise ValueError(f"PyPI package identifier must be {PYPI_PACKAGE!r}")
    if pypi.get("version") != version:
        raise ValueError(
            f"PyPI package version {pypi.get('version')!r} does not match {version!r}"
        )
    if pypi.get("transport", {}).get("type") != "stdio":
        raise ValueError("PyPI package transport must be 'stdio'")

    oci = _one_package(server, "oci")
    expected_oci = f"{OCI_REPOSITORY}:v{version}"
    if oci.get("identifier") != expected_oci:
        raise ValueError(
            f"OCI package identifier {oci.get('identifier')!r} does not match "
            f"{expected_oci!r}"
        )

    marker = f"mcp-name: {SERVER_NAME}"
    if marker not in readme:
        raise ValueError(f"README.md must contain the MCP ownership marker {marker!r}")
    source_label = f'org.opencontainers.image.source="{GITHUB_URL}"'
    if source_label not in dockerfile:
        raise ValueError(f"Dockerfile must contain the OCI source label {source_label!r}")
    server_label = f'io.modelcontextprotocol.server.name="{SERVER_NAME}"'
    if server_label not in dockerfile:
        raise ValueError(f"Dockerfile must contain the MCP server label {server_label!r}")

    requires_python = str(project.get("requires-python", ""))
    if not re.fullmatch(r">=3\.12", requires_python):
        raise ValueError(
            "project.requires-python must match the CI-supported range "
            "'>=3.12'"
        )
    return version


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--expected-version",
        help="Release tag or version that pyproject.toml must match.",
    )
    args = parser.parse_args()
    try:
        version = check(args.expected_version)
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(f"release metadata check failed: {exc}", file=sys.stderr)
        return 1
    print(f"release metadata is aligned at {version}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

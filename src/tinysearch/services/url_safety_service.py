"""URL safety checks shared by /scrape and the MCP scrape_url tool.

Enforces the SSRF hardening required by upstream issue #10:

- Only ``http`` and ``https`` schemes.
- Reject URLs with embedded credentials.
- Reject IP literals and resolved addresses that are loopback, private,
  link-local, multicast, reserved or unspecified.
- DNS rebinding mitigation: reject the URL if ANY resolved address is
  non-public, not just at least one.
- Apply configured ``blocked_domains`` to both the initial URL and any
  redirect target the caller re-validates.
"""

from __future__ import annotations

import socket
from collections.abc import Iterable
from ipaddress import AddressValueError, ip_address
from urllib.parse import urlsplit

from services.web_search_service import is_blocked_domain


_ALLOWED_SCHEMES = frozenset({"http", "https"})


class InvalidUrlError(ValueError):
    """URL is malformed, missing host, or uses a disallowed scheme."""


class BlockedUrlError(ValueError):
    """URL resolves to a non-public address or matches a blocked domain."""


def _assert_public_ip(addr, *, source: str) -> None:
    if (
        addr.is_loopback
        or addr.is_private
        or addr.is_link_local
        or addr.is_multicast
        or addr.is_reserved
        or addr.is_unspecified
    ):
        raise BlockedUrlError(f"address is not publicly routable: {source}")


def validate_public_url(url: str) -> str:
    """Validate scheme, structure and IP literal. Does not resolve DNS.

    Returns the cleaned URL string. Raises ``InvalidUrlError`` for malformed
    URLs or disallowed schemes, and ``BlockedUrlError`` when the host is a
    non-public IP literal.
    """
    if not isinstance(url, str) or not url.strip():
        raise InvalidUrlError("URL must be a non-empty string")
    cleaned = url.strip()
    parsed = urlsplit(cleaned)
    scheme = parsed.scheme.lower()
    if scheme not in _ALLOWED_SCHEMES:
        raise InvalidUrlError(
            f"only http and https URLs are allowed, got {scheme or 'no scheme'!r}"
        )
    if parsed.username or parsed.password:
        raise InvalidUrlError("URLs with embedded credentials are not allowed")
    host = parsed.hostname
    if not host:
        raise InvalidUrlError("URL has no host")
    try:
        addr = ip_address(host)
    except (ValueError, AddressValueError):
        return cleaned
    _assert_public_ip(addr, source=f"host literal {host!r}")
    return cleaned


def resolve_and_check_public(host: str) -> list[str]:
    """Resolve ``host`` and reject if ANY resolved address is non-public.

    Returns the list of resolved address strings on success.
    """
    if not host:
        raise InvalidUrlError("hostname is empty")
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror as exc:
        raise InvalidUrlError(f"could not resolve host {host!r}: {exc}") from exc
    resolved: list[str] = []
    for info in infos:
        sockaddr = info[4]
        addr_text = sockaddr[0]
        try:
            addr = ip_address(addr_text)
        except (ValueError, AddressValueError) as exc:
            raise BlockedUrlError(
                f"resolver returned non-IP address for {host!r}: {addr_text!r}"
            ) from exc
        _assert_public_ip(addr, source=f"{host!r} -> {addr_text}")
        resolved.append(addr_text)
    if not resolved:
        raise InvalidUrlError(f"no addresses resolved for {host!r}")
    return resolved


def enforce_blocked_domains(url: str, blocked_domains: Iterable[str]) -> None:
    if is_blocked_domain(url, blocked_domains):
        raise BlockedUrlError(
            f"URL host is in the configured blocked-domains list: {url}"
        )


def assert_url_is_fetchable(url: str, blocked_domains: Iterable[str]) -> str:
    """Run scheme, credential, IP-literal, blocked-domain and DNS checks.

    Returns the validated URL string. Hosts that are already IP literals skip
    the DNS resolution step (the literal itself was already checked).
    """
    validated = validate_public_url(url)
    enforce_blocked_domains(validated, blocked_domains)
    host = urlsplit(validated).hostname or ""
    try:
        ip_address(host)
    except (ValueError, AddressValueError):
        resolve_and_check_public(host)
    return validated

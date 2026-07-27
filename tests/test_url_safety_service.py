from __future__ import annotations

import socket
import unittest
from unittest.mock import patch

from tinysearch.services.url_safety_service import (
    BlockedUrlError,
    InvalidUrlError,
    assert_url_is_fetchable,
    enforce_blocked_domains,
    resolve_and_check_public,
    validate_public_url,
)


def _addrinfo(ip: str, *, family: int = socket.AF_INET) -> tuple:
    port = 0
    if family == socket.AF_INET6:
        sockaddr = (ip, port, 0, 0)
    else:
        sockaddr = (ip, port)
    return (family, socket.SOCK_STREAM, 6, "", sockaddr)


class ValidatePublicUrlTests(unittest.TestCase):
    def test_accepts_https_url(self) -> None:
        self.assertEqual(
            validate_public_url("https://example.com/a"),
            "https://example.com/a",
        )

    def test_accepts_http_url(self) -> None:
        self.assertEqual(
            validate_public_url("http://example.com"),
            "http://example.com",
        )

    def test_strips_surrounding_whitespace(self) -> None:
        self.assertEqual(
            validate_public_url("  https://example.com/a  "),
            "https://example.com/a",
        )

    def test_rejects_empty_string(self) -> None:
        with self.assertRaises(InvalidUrlError):
            validate_public_url("")

    def test_rejects_file_scheme(self) -> None:
        with self.assertRaises(InvalidUrlError):
            validate_public_url("file:///etc/passwd")

    def test_rejects_ftp_scheme(self) -> None:
        with self.assertRaises(InvalidUrlError):
            validate_public_url("ftp://example.com/x")

    def test_rejects_javascript_scheme(self) -> None:
        with self.assertRaises(InvalidUrlError):
            validate_public_url("javascript:alert(1)")

    def test_rejects_userinfo(self) -> None:
        with self.assertRaises(InvalidUrlError):
            validate_public_url("https://user:password@example.com/x")

    def test_rejects_missing_host(self) -> None:
        with self.assertRaises(InvalidUrlError):
            validate_public_url("https:///x")

    def test_rejects_ipv4_loopback_literal(self) -> None:
        with self.assertRaises(BlockedUrlError):
            validate_public_url("http://127.0.0.1/x")

    def test_rejects_ipv6_loopback_literal(self) -> None:
        with self.assertRaises(BlockedUrlError):
            validate_public_url("http://[::1]/x")

    def test_rejects_private_ipv4_literal(self) -> None:
        with self.assertRaises(BlockedUrlError):
            validate_public_url("http://10.0.0.1/x")
        with self.assertRaises(BlockedUrlError):
            validate_public_url("http://192.168.1.1/x")

    def test_rejects_link_local_ipv4_literal(self) -> None:
        with self.assertRaises(BlockedUrlError):
            validate_public_url("http://169.254.0.1/x")

    def test_rejects_multicast_ipv4_literal(self) -> None:
        with self.assertRaises(BlockedUrlError):
            validate_public_url("http://224.0.0.1/x")


class ResolveAndCheckPublicTests(unittest.TestCase):
    def test_accepts_public_ipv4(self) -> None:
        with patch(
            "tinysearch.services.url_safety_service.socket.getaddrinfo",
            return_value=[_addrinfo("93.184.216.34")],
        ):
            self.assertEqual(
                resolve_and_check_public("example.com"),
                ["93.184.216.34"],
            )

    def test_rejects_when_any_resolved_address_is_private(self) -> None:
        with patch(
            "tinysearch.services.url_safety_service.socket.getaddrinfo",
            return_value=[
                _addrinfo("93.184.216.34"),
                _addrinfo("10.0.0.1"),
            ],
        ):
            with self.assertRaises(BlockedUrlError):
                resolve_and_check_public("example.com")

    def test_rejects_when_resolved_to_loopback_only(self) -> None:
        with patch(
            "tinysearch.services.url_safety_service.socket.getaddrinfo",
            return_value=[_addrinfo("127.0.0.1")],
        ):
            with self.assertRaises(BlockedUrlError):
                resolve_and_check_public("localhost")

    def test_propagates_dns_failure_as_invalid_url(self) -> None:
        with patch(
            "tinysearch.services.url_safety_service.socket.getaddrinfo",
            side_effect=socket.gaierror("not found"),
        ):
            with self.assertRaises(InvalidUrlError):
                resolve_and_check_public("nope.invalid")


class EnforceBlockedDomainsTests(unittest.TestCase):
    def test_rejects_exact_host_match(self) -> None:
        with self.assertRaises(BlockedUrlError):
            enforce_blocked_domains(
                "https://blocked.example/x",
                ["blocked.example"],
            )

    def test_rejects_subdomain_of_blocked_host(self) -> None:
        with self.assertRaises(BlockedUrlError):
            enforce_blocked_domains(
                "https://news.blocked.example/x",
                ["blocked.example"],
            )

    def test_allows_unrelated_host(self) -> None:
        enforce_blocked_domains(
            "https://allowed.example/x",
            ["blocked.example"],
        )


class AssertUrlIsFetchableTests(unittest.TestCase):
    def test_happy_path_with_public_dns(self) -> None:
        with patch(
            "tinysearch.services.url_safety_service.socket.getaddrinfo",
            return_value=[_addrinfo("93.184.216.34")],
        ):
            self.assertEqual(
                assert_url_is_fetchable(
                    "https://example.com/a",
                    blocked_domains=[],
                ),
                "https://example.com/a",
            )

    def test_blocked_domain_short_circuits_before_dns(self) -> None:
        with patch(
            "tinysearch.services.url_safety_service.socket.getaddrinfo"
        ) as getaddrinfo:
            with self.assertRaises(BlockedUrlError):
                assert_url_is_fetchable(
                    "https://blocked.example/x",
                    blocked_domains=["blocked.example"],
                )
            getaddrinfo.assert_not_called()

    def test_invalid_scheme_short_circuits(self) -> None:
        with self.assertRaises(InvalidUrlError):
            assert_url_is_fetchable("ftp://example.com/x", blocked_domains=[])

    def test_private_resolved_address_is_blocked(self) -> None:
        with patch(
            "tinysearch.services.url_safety_service.socket.getaddrinfo",
            return_value=[_addrinfo("10.0.0.1")],
        ):
            with self.assertRaises(BlockedUrlError):
                assert_url_is_fetchable(
                    "https://internal.example/x",
                    blocked_domains=[],
                )

    def test_ip_literal_skips_dns_resolution(self) -> None:
        with patch(
            "tinysearch.services.url_safety_service.socket.getaddrinfo"
        ) as getaddrinfo:
            self.assertEqual(
                assert_url_is_fetchable(
                    "https://93.184.216.34/a",
                    blocked_domains=[],
                ),
                "https://93.184.216.34/a",
            )
            getaddrinfo.assert_not_called()


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import unittest
from inspect import signature
from unittest.mock import AsyncMock, patch

from fastapi import HTTPException
from pydantic import ValidationError

from tinysearch.services.browser_tool_service import (
    BrowserDisabledError,
    BrowserToolError,
)
from tinysearch.servers.fastapi_server import (
    BrowserActRequest,
    BrowserNavigateRequest,
    app,
    browser_act_endpoint,
    browser_navigate_endpoint,
)
from tinysearch.servers.mcp_server import browser_act_tool, browser_navigate_tool


def _tool_function(tool):
    return getattr(tool, "fn", tool)


class BrowserRequestValidationTests(unittest.TestCase):
    def test_fastapi_exposes_both_mcp_browser_surfaces(self) -> None:
        paths = {route.path for route in app.routes}
        self.assertIn("/browser/navigate", paths)
        self.assertIn("/browser/act", paths)

    def test_fastapi_request_fields_mirror_mcp_tool_arguments(self) -> None:
        self.assertEqual(
            list(BrowserNavigateRequest.model_fields),
            list(signature(_tool_function(browser_navigate_tool)).parameters),
        )
        self.assertEqual(
            list(BrowserActRequest.model_fields),
            list(signature(_tool_function(browser_act_tool)).parameters),
        )

    def test_screenshot_parameters_are_not_part_of_the_contract(self) -> None:
        with self.assertRaises(ValidationError):
            BrowserActRequest(action="look", full_page=True)


class BrowserEndpointTests(unittest.IsolatedAsyncioTestCase):
    async def test_navigate_uses_the_shared_browser_service(self) -> None:
        call_tool = AsyncMock(return_value="- Page URL: https://example.com/")
        config = {"browser_backend": "playwright"}
        with patch(
            "tinysearch.servers.fastapi_server.load_tinysearch_config",
            return_value=config,
        ), patch(
            "tinysearch.servers.fastapi_server.browser_tool_service.call_tool",
            call_tool,
        ):
            payload = await browser_navigate_endpoint(
                BrowserNavigateRequest(
                    url="https://example.com",
                    find="pricing",
                    depth=3,
                )
            )

        self.assertEqual(payload, {"result": "- Page URL: https://example.com/"})
        call_tool.assert_awaited_once_with(
            "navigate",
            config,
            url="https://example.com/",
            find="pricing",
            depth=3,
        )

    async def test_act_uses_the_same_folded_arguments_as_mcp(self) -> None:
        call_tool = AsyncMock(return_value="Found 1 matches")
        with patch(
            "tinysearch.servers.fastapi_server.load_tinysearch_config",
            return_value={"browser_backend": "playwright"},
        ), patch(
            "tinysearch.servers.fastapi_server.browser_tool_service.call_tool",
            call_tool,
        ):
            payload = await browser_act_endpoint(
                BrowserActRequest(
                    action="wait_for",
                    text="Ready",
                    find="Results",
                )
            )

        self.assertEqual(payload, {"result": "Found 1 matches"})
        self.assertEqual(call_tool.await_args.args[0], "wait_for")
        self.assertEqual(
            call_tool.await_args.kwargs,
            {"text": "Ready", "find": "Results"},
        )

    async def test_invalid_action_is_an_explicit_client_error(self) -> None:
        with self.assertRaises(HTTPException) as raised:
            await browser_act_endpoint(BrowserActRequest(action="take_screenshot"))

        self.assertEqual(raised.exception.status_code, 400)
        self.assertEqual(raised.exception.detail["code"], "browser_error")

    async def test_disabled_browser_is_service_unavailable(self) -> None:
        with patch(
            "tinysearch.servers.fastapi_server.browser_tool_service.call_tool",
            new=AsyncMock(side_effect=BrowserDisabledError("browser disabled")),
        ):
            with self.assertRaises(HTTPException) as raised:
                await browser_navigate_endpoint(
                    BrowserNavigateRequest(url="https://example.com")
                )

        self.assertEqual(raised.exception.status_code, 503)
        self.assertEqual(raised.exception.detail["code"], "browser_disabled")

    async def test_browser_failure_is_an_explicit_client_error(self) -> None:
        with patch(
            "tinysearch.servers.fastapi_server.browser_tool_service.call_tool",
            new=AsyncMock(side_effect=BrowserToolError("navigation failed")),
        ):
            with self.assertRaises(HTTPException) as raised:
                await browser_navigate_endpoint(
                    BrowserNavigateRequest(url="https://example.com")
                )

        self.assertEqual(raised.exception.status_code, 400)
        self.assertEqual(raised.exception.detail["code"], "browser_error")


if __name__ == "__main__":
    unittest.main()

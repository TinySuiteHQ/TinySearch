from __future__ import annotations

import unittest
from unittest.mock import AsyncMock, patch

from pydantic import ValidationError

from tinysearch.servers.fastapi_server import BrowseRequest, app, browse_endpoint


class BrowseRequestValidationTests(unittest.TestCase):
    def test_requires_url_or_session_id(self) -> None:
        with self.assertRaises(ValidationError):
            BrowseRequest()

    def test_accepts_session_id_without_url(self) -> None:
        request = BrowseRequest(session_id="sess-1", actions=[{"action": "click", "selector": "#x"}])
        self.assertIsNone(request.url)
        self.assertEqual(request.actions[0].selector, "#x")

    def test_rejects_unknown_action_field(self) -> None:
        with self.assertRaises(ValidationError):
            BrowseRequest(url="https://example.com", actions=[{"action": "hover"}])

    def test_exposes_browse_route(self) -> None:
        paths = {route.path for route in app.routes}
        self.assertIn("/browse", paths)


class BrowseEndpointTests(unittest.IsolatedAsyncioTestCase):
    async def test_passes_url_actions_and_session_id_through_to_core(self) -> None:
        browse_mock = AsyncMock(return_value={"operation": "browse", "status": "ok"})
        with patch("tinysearch.core.browse", browse_mock):
            payload = await browse_endpoint(
                BrowseRequest(
                    url="https://example.com",
                    actions=[{"action": "click", "selector": "#accept"}],
                    query="pricing",
                    max_tokens=321,
                )
            )

        self.assertEqual(payload["operation"], "browse")
        self.assertEqual(browse_mock.await_args.args[0], "https://example.com/")
        self.assertEqual(browse_mock.await_args.args[1], [{"action": "click", "selector": "#accept"}])
        self.assertEqual(browse_mock.await_args.kwargs["query"], "pricing")
        self.assertEqual(browse_mock.await_args.kwargs["max_tokens"], 321)

    async def test_omits_url_when_continuing_a_session(self) -> None:
        browse_mock = AsyncMock(return_value={"operation": "browse", "status": "ok"})
        with patch("tinysearch.core.browse", browse_mock):
            await browse_endpoint(BrowseRequest(session_id="sess-1"))

        self.assertIsNone(browse_mock.await_args.args[0])
        self.assertEqual(browse_mock.await_args.kwargs["session_id"], "sess-1")


if __name__ == "__main__":
    unittest.main()

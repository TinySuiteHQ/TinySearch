from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType
from unittest.mock import AsyncMock, MagicMock, patch

from tinysearch.services.browser_runtime_service import PlaywrightRuntime


class PlaywrightRuntimeTests(unittest.IsolatedAsyncioTestCase):
    async def test_local_start_launches_owned_chromium(self) -> None:
        browser = MagicMock(close=AsyncMock())
        chromium = MagicMock()
        chromium.launch = AsyncMock(return_value=browser)
        chromium.connect_over_cdp = AsyncMock()
        driver = MagicMock(chromium=chromium)
        driver.stop = AsyncMock()
        manager = MagicMock()
        manager.start = AsyncMock(return_value=driver)

        playwright_pkg = ModuleType("playwright")
        async_api = ModuleType("playwright.async_api")
        async_api.async_playwright = lambda: manager

        runtime = PlaywrightRuntime({})
        with patch.dict(
            sys.modules,
            {"playwright": playwright_pkg, "playwright.async_api": async_api},
        ):
            await runtime.start()

        chromium.launch.assert_awaited_once_with(headless=True)
        chromium.connect_over_cdp.assert_not_awaited()
        await runtime.close()
        browser.close.assert_awaited_once()
        driver.stop.assert_awaited_once()

    async def test_cdp_start_attaches_without_owning_remote_browser(self) -> None:
        browser = MagicMock(close=AsyncMock())
        chromium = MagicMock()
        chromium.launch = AsyncMock()
        chromium.connect_over_cdp = AsyncMock(return_value=browser)
        driver = MagicMock(chromium=chromium)
        driver.stop = AsyncMock()
        manager = MagicMock()
        manager.start = AsyncMock(return_value=driver)

        playwright_pkg = ModuleType("playwright")
        async_api = ModuleType("playwright.async_api")
        async_api.async_playwright = lambda: manager

        runtime = PlaywrightRuntime({"browser_cdp_url": "http://browser:9222"})
        with patch.dict(
            sys.modules,
            {"playwright": playwright_pkg, "playwright.async_api": async_api},
        ):
            await runtime.start()

        chromium.connect_over_cdp.assert_awaited_once_with("http://browser:9222")
        chromium.launch.assert_not_awaited()
        await runtime.close()
        browser.close.assert_not_awaited()
        driver.stop.assert_awaited_once()

    async def test_new_context_seeds_storage_state_and_applies_action_timeout(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state = Path(tmp) / "state.json"
            state.write_text("{}", encoding="utf-8")
            context = MagicMock()
            browser = MagicMock()
            browser.new_context = AsyncMock(return_value=context)

            runtime = PlaywrightRuntime(
                {
                    "browser_storage_state_path": str(state),
                    "browser_action_timeout_seconds": 2.5,
                }
            )
            runtime._browser = browser

            result = await runtime.new_context(apply_action_timeout=True)

        self.assertIs(result, context)
        browser.new_context.assert_awaited_once_with(
            locale="en-US",
            storage_state=str(state),
        )
        context.set_default_timeout.assert_called_once_with(2500.0)

    async def test_new_context_applies_configured_proxy(self) -> None:
        context = MagicMock()
        browser = MagicMock()
        browser.new_context = AsyncMock(return_value=context)

        runtime = PlaywrightRuntime(
            {
                "browser_proxy_server": "http://proxy.example.com:8000",
                "browser_proxy_username": "user",
                "browser_proxy_password": "pass",
                "browser_proxy_bypass": "internal.example.com",
            }
        )
        runtime._browser = browser

        await runtime.new_context()

        browser.new_context.assert_awaited_once_with(
            locale="en-US",
            proxy={
                "server": "http://proxy.example.com:8000",
                "username": "user",
                "password": "pass",
                "bypass": "internal.example.com",
            },
        )

    async def test_new_context_round_robins_across_multiple_proxies(self) -> None:
        context = MagicMock()
        browser = MagicMock()
        browser.new_context = AsyncMock(return_value=context)

        runtime = PlaywrightRuntime(
            {"browser_proxy_server": "http://proxy-a:8000, http://proxy-b:8000"}
        )
        runtime._browser = browser

        await runtime.new_context()
        await runtime.new_context()
        await runtime.new_context()

        servers = [
            call.kwargs["proxy"]["server"]
            for call in browser.new_context.await_args_list
        ]
        self.assertEqual(
            servers,
            ["http://proxy-a:8000", "http://proxy-b:8000", "http://proxy-a:8000"],
        )

    async def test_persist_storage_state_creates_parent_and_writes_context(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state = Path(tmp) / "nested" / "state.json"
            context = MagicMock()
            context.storage_state = AsyncMock()

            runtime = PlaywrightRuntime({"browser_storage_state_path": str(state)})
            await runtime.persist_storage_state(context)

            self.assertTrue(state.parent.is_dir())
            context.storage_state.assert_awaited_once_with(path=str(state))


if __name__ == "__main__":
    unittest.main()

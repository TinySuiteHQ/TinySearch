"""Opt-in activity, the tinysuite.manage/1 views, and who may read which."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch
from urllib.parse import quote, urlencode

from pydantic import AnyUrl

try:
    from tests.mcp_http_harness import RunningMcpServer, mcp_session
except ImportError:  # discovered with tests/ as the top-level directory
    from mcp_http_harness import RunningMcpServer, mcp_session

from tinysearch.services import activity_service as activity
from tinysearch.services import management_service as manage
from tinysearch.servers import mcp_server

HEADER = "X-TinySearch-User-Id"
TOKEN = "k" * 32


def search_result(query: str, urls: list[str]) -> dict:
    return {
        "status": "ok",
        "items": [{
            "query": query, "status": "ok", "error": None,
            "results": [{"rank": i, "url": u, "title": "", "preview": ""} for i, u in enumerate(urls, 1)],
            "domains": [], "backend_attempts": [], "stats": {},
        }],
        "stats": {"search_item_count": 1, "backend_attempt_count": 1},
    }


def scrape_result(ok: list[str], failed: dict[str, str] | None = None) -> dict:
    results = [{"status": "ok", "result": {"query": "*", "sources": [{"url": u}], "stats": {"requested_url": u}}}
               for u in ok]
    results += [{"status": "error", "url": u, "query": "*", "error": {"code": code, "message": ""}}
                for u, code in (failed or {}).items()]
    return {"status": "partial" if failed else "ok", "results": results}


def view(name: str, **params: object) -> str:
    raw = quote(urlencode(params), safe="") if params else "-"
    return manage.VIEW_URI.format(view=name).format(params=raw)


async def read(session, uri: str) -> dict:  # noqa: ANN001
    result = await session.read_resource(AnyUrl(uri))
    return json.loads(result.contents[0].text)


class _ActivityBase(unittest.TestCase):
    mode = "history"

    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.db = Path(self._tmpdir.name) / "activity.db"
        env = patch.dict(os.environ, {"TINYSEARCH_ACTIVITY": self.mode, "TINYSEARCH_ACTIVITY_DB": str(self.db)})
        env.start()
        self.addCleanup(env.stop)
        activity.reset_store()
        self.addCleanup(activity.reset_store)

    def tearDown(self) -> None:
        activity.reset_store()
        self._tmpdir.cleanup()


class OffByDefaultTests(_ActivityBase):
    mode = "off"

    def test_nothing_is_written_or_even_created(self) -> None:
        activity.record_quietly("search", search_result("q", ["https://a.example/x"]), principal="alice", latency_ms=5)
        self.assertFalse(self.db.exists())

    def test_the_default_is_off(self) -> None:
        with patch.dict(os.environ, {"TINYSEARCH_ACTIVITY": ""}):
            self.assertEqual(activity.activity_mode(), "off")


class AggregateTests(_ActivityBase):
    mode = "aggregate"

    def test_counts_domains_but_keeps_no_queries_urls_or_callers(self) -> None:
        activity.record_search(search_result("secret query", ["https://www.a.example/1", "https://b.example/2"]),
                               principal="alice", latency_ms=10)
        activity.record_scrape(scrape_result(["https://a.example/1"], {"https://c.example/x": "BlockedUrlError"}),
                               principal="alice", latency_ms=20)
        store = activity.get_store()
        rows = store._rows("SELECT principal, query FROM events")
        self.assertTrue(all(r["principal"] is None and r["query"] is None for r in rows))
        self.assertEqual(store._rows("SELECT COUNT(*) AS n FROM urls WHERE url IS NOT NULL")[0]["n"], 0)
        domains = {r["domain"]: r for r in store.domains(search=None, source=None, offset=0, limit=50)["rows"]}
        self.assertEqual((domains["a.example"]["discovered"], domains["a.example"]["fetched"]), (1, 1))
        self.assertEqual(domains["c.example"]["failed"], 1)
        self.assertEqual(domains["b.example"]["fetched"], 0)


class HistoryTests(_ActivityBase):
    def test_history_is_per_caller_hashed_and_unattributed_activity_stays_aggregate(self) -> None:
        activity.record_search(search_result("alice q", ["https://a.example/1"]), principal="alice", latency_ms=1)
        activity.record_search(search_result("bob q", ["https://b.example/1"]), principal="bob", latency_ms=1)
        activity.record_search(search_result("anon q", ["https://c.example/1"]), principal=None, latency_ms=1)
        store = activity.get_store()
        alice = store.events_for("alice", search=None, offset=0, limit=50)
        self.assertEqual([r["query"] for r in alice["rows"]], ["alice q"])
        self.assertEqual([r["url"] for r in store.urls_for("bob", source=None, search=None, offset=0, limit=50)["rows"]],
                         ["https://b.example/1"])
        stored = {r["principal"] for r in store._rows("SELECT principal FROM events")}
        self.assertNotIn("alice", stored, "callers are stored hashed")
        self.assertEqual(store._rows("SELECT COUNT(*) AS n FROM events WHERE query = 'anon q'")[0]["n"], 0)
        self.assertEqual(store.domains(search="c.example", source=None, offset=0, limit=5)["total"], 1)

    def test_clear_removes_only_that_callers_rows(self) -> None:
        for who in ("alice", "bob"):
            activity.record_search(search_result(f"{who} q", ["https://x.example"]), principal=who, latency_ms=1)
        self.assertEqual(activity.get_store().clear("alice"), 1)
        store = activity.get_store()
        self.assertEqual(store.events_for("alice", search=None, offset=0, limit=5)["total"], 0)
        self.assertEqual(store.events_for("bob", search=None, offset=0, limit=5)["total"], 1)

    def test_retention_and_the_event_cap(self) -> None:
        store = activity.get_store()
        for i in range(5):
            activity.record_search(search_result(f"q{i}", []), principal="alice", latency_ms=1)
        store._conn.execute("UPDATE events SET ts = '2000-01-01T00:00:00Z' WHERE query = 'q0'")
        store._conn.commit()
        store.purge(days=7, keep=3)
        self.assertEqual([r["query"] for r in store.events_for("alice", search=None, offset=0, limit=9)["rows"]],
                         ["q4", "q3", "q2"])

    def test_bad_settings_are_rejected(self) -> None:
        for name, value in (("TINYSEARCH_ACTIVITY", "everything"), ("TINYSEARCH_ACTIVITY_RETENTION_DAYS", "0"),
                            ("TINYSEARCH_ACTIVITY_MAX_EVENTS", "lots")):
            with self.subTest(name=name), patch.dict(os.environ, {name: value}), self.assertRaises(ValueError):
                manage.check_management_startup("stdio")


class StartupTests(unittest.TestCase):
    def test_http_management_needs_a_token_or_a_trusted_identity(self) -> None:
        base = {"TINYSEARCH_MANAGEMENT": "on", "TINYSEARCH_MANAGEMENT_TOKEN": "",
                "TINYSEARCH_TRUSTED_USER_HEADER": "", "TINYSEARCH_TRUSTED_PROXY_CIDRS": ""}
        with patch.dict(os.environ, base):
            with self.assertRaises(ValueError):
                manage.check_management_startup("streamable-http")
            manage.check_management_startup("stdio")
        with patch.dict(os.environ, {**base, "TINYSEARCH_MANAGEMENT_TOKEN": TOKEN}):
            manage.check_management_startup("sse")
        with patch.dict(os.environ, {**base, "TINYSEARCH_TRUSTED_USER_HEADER": HEADER}):
            with self.assertRaises(ValueError):  # a header alone is never believed
                manage.check_management_startup("streamable-http")

    def test_an_http_message_without_its_request_is_nobody(self) -> None:
        # If the SDK ever stops attaching the HTTP request, a caller must not
        # become the local operator: that is decided by the configured transport.
        with patch.dict(os.environ, {"MCP_TRANSPORT": "streamable-http", "TINYSEARCH_MANAGEMENT": "on",
                                     "TINYSEARCH_MANAGEMENT_TOKEN": TOKEN}),                 patch.object(manage, "_mcp_request", return_value=(True, None)):
            self.assertIsNone(manage.current_principal())
            self.assertFalse(manage._instance_allowed())
            with self.assertRaises(manage.ManagementAccessError):
                manage.authorize("instance")
        with patch.dict(os.environ, {"MCP_TRANSPORT": "stdio"}),                 patch.object(manage, "_mcp_request", return_value=(True, None)):
            self.assertEqual(manage.current_principal(), manage.LOCAL_PRINCIPAL)

    def test_identity_is_believed_only_from_trusted_peers_and_only_once(self) -> None:
        env = {"TINYSEARCH_TRUSTED_USER_HEADER": HEADER, "TINYSEARCH_TRUSTED_PROXY_CIDRS": "10.0.0.0/8"}
        header = HEADER.lower().encode()
        with patch.dict(os.environ, env):
            ok = {"client": ("10.1.2.3", 1), "headers": [(header, b"alice")]}
            self.assertEqual(manage.principal_from_scope(ok), "alice")
            self.assertIsNone(manage.principal_from_scope({**ok, "client": ("192.168.1.5", 1)}))
            self.assertIsNone(manage.principal_from_scope({**ok, "headers": [(header, b"a"), (header, b"b")]}))
            self.assertIsNone(manage.principal_from_scope({**ok, "headers": [(header, b"a\nb")]}))
            self.assertIsNone(manage.principal_from_scope({**ok, "headers": []}))


class McpManagementTests(unittest.IsolatedAsyncioTestCase):
    """The real streamable-HTTP app, the real MCP client, stubbed search backends."""

    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        env = patch.dict(os.environ, {
            "TINYSEARCH_ACTIVITY": "history",
            "TINYSEARCH_ACTIVITY_DB": str(Path(self._tmpdir.name) / "activity.db"),
            "TINYSEARCH_MANAGEMENT": "on",
            "TINYSEARCH_MANAGEMENT_TOKEN": TOKEN,
            "TINYSEARCH_TRUSTED_USER_HEADER": HEADER,
            "TINYSEARCH_TRUSTED_PROXY_CIDRS": "127.0.0.0/8",
        })
        env.start()
        self.addCleanup(env.stop)
        activity.reset_store()
        self.addCleanup(activity.reset_store)
        # Registration is process-wide; put the server back as other tests expect it.
        tools = dict(mcp_server.mcp._tool_manager._tools)
        resources = dict(mcp_server.mcp._resource_manager._resources)
        templates = dict(mcp_server.mcp._resource_manager._templates)

        def restore() -> None:
            mcp_server.mcp._tool_manager._tools = tools
            mcp_server.mcp._resource_manager._resources = resources
            mcp_server.mcp._resource_manager._templates = templates
            mcp_server._management_registered = False

        self.addCleanup(restore)
        mcp_server._management_registered = False
        mcp_server.register_management()

        async def fake_search(items, **_kwargs):  # noqa: ANN001
            query = items[0]["query"]
            return search_result(query, [f"https://{query.split()[0]}.example/page"])

        for target, value in (("tinysearch.core.search", AsyncMock(side_effect=fake_search)),
                              ("tinysearch.core.scrape_urls", AsyncMock(return_value=scrape_result(["https://shared.example/doc"])))):
            patcher = patch(target, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def tearDown(self) -> None:
        activity.reset_store()
        self._tmpdir.cleanup()

    async def _search_as(self, server, user: str) -> None:  # noqa: ANN001
        async with mcp_session(server.url, {HEADER: user}) as session:
            await session.call_tool("search", {"items": [{"query": f"{user} private topic"}]})
            await session.call_tool("scrape_urls", {"items": [{"url": "https://shared.example/doc"}]})

    async def test_each_caller_reads_only_their_own_history(self) -> None:
        with RunningMcpServer() as server:
            await self._search_as(server, "alice")
            await self._search_as(server, "bob")
            async with mcp_session(server.url, {HEADER: "alice"}) as session:
                mine = await read(session, view("activity"))
                pages = await read(session, view("pages", source="discovered"))
                manifest = await read(session, manage.MANIFEST_URI)
                with self.assertRaises(Exception):  # instance views need the token
                    await read(session, view("domains"))

        self.assertEqual({r["query"] for r in mine["rows"] if r["query"]}, {"alice private topic"})
        self.assertEqual([r["url"] for r in pages["rows"]], ["https://alice.example/page"])
        self.assertEqual([v["id"] for v in manifest["views"]], ["activity", "pages"])

    async def test_the_token_opens_aggregate_views_that_show_no_queries(self) -> None:
        with RunningMcpServer() as server:
            await self._search_as(server, "alice")
            await self._search_as(server, "bob")
            async with mcp_session(server.url, {"Authorization": f"Bearer {TOKEN}"}) as session:
                manifest = await read(session, manage.MANIFEST_URI)
                domains = await read(session, view("domains"))
                overview = await read(session, view("overview"))
                rules = await read(session, view("rules"))
                with self.assertRaises(Exception):  # no identity: no "own" history
                    await read(session, view("activity"))

        self.assertEqual([v["id"] for v in manifest["views"]], ["overview", "domains", "rules"])
        by_domain = {r["domain"]: r for r in domains["rows"]}
        self.assertEqual(by_domain["shared.example"]["fetched"], 2)
        self.assertEqual(by_domain["alice.example"]["discovered"], 1)
        self.assertNotIn("private topic", json.dumps(domains) + json.dumps(overview))
        labels = {i["label"]: i["value"] for i in overview["items"]}
        self.assertEqual(labels["Searches, last 24h"], 2)
        self.assertTrue(any(r["source"] == "built in" for r in rules["rows"]))

    async def test_a_forged_identity_from_an_untrusted_peer_reads_nothing(self) -> None:
        with RunningMcpServer() as server:
            await self._search_as(server, "alice")
            with patch.dict(os.environ, {"TINYSEARCH_TRUSTED_PROXY_CIDRS": "10.0.0.0/8"}):
                async with mcp_session(server.url, {HEADER: "alice"}) as session:
                    with self.assertRaises(Exception):
                        await read(session, view("activity"))

    async def test_clear_tool_deletes_only_the_callers_history(self) -> None:
        with RunningMcpServer() as server:
            await self._search_as(server, "alice")
            await self._search_as(server, "bob")
            async with mcp_session(server.url, {HEADER: "alice"}) as session:
                await session.call_tool("clear_search_history", {})
                mine = await read(session, view("activity"))
            async with mcp_session(server.url, {HEADER: "bob"}) as session:
                theirs = await read(session, view("activity"))
        self.assertEqual(mine["total"], 0)
        self.assertEqual(theirs["total"], 2)


if __name__ == "__main__":
    unittest.main()

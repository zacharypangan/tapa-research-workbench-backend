"""Bounded guard tests and an explicitly enabled, owned localhost browser fixture.

Run integration with TAPA_TEST_BROWSER=selenium|playwright. Selenium requires an
existing TAPA_TEST_DRIVER; these tests never install a driver or invoke Manager.
"""

from contextlib import ExitStack
from http.client import HTTPConnection
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import os
from pathlib import Path
import socket
import threading
import unittest
from unittest.mock import patch

try:
    from . import browser_engines
except ImportError:
    import browser_engines


def dns_record(address):
    family = socket.AF_INET6 if ":" in address else socket.AF_INET
    return (family, socket.SOCK_STREAM, 6, "", (address, 443))


class PublicURLTests(unittest.TestCase):
    def test_public_url_normalization_and_exact_host_scope(self):
        with patch.object(browser_engines.socket, "getaddrinfo", return_value=[dns_record("93.184.216.34")]) as dns:
            self.assertEqual(
                browser_engines.validate_public_url("https://EXAMPLE.com", {"example.com"}),
                "https://example.com/",
            )
            dns.assert_called_once()
        for url in ("https://example.com.evil.test/", "https://www.example.com/"):
            with self.subTest(url=url), patch.object(browser_engines.socket, "getaddrinfo") as dns:
                with self.assertRaises(ValueError):
                    browser_engines.validate_public_url(url, {"example.com"})
                dns.assert_not_called()

    def test_unsafe_schemes_credentials_and_malformed_urls(self):
        for url in (
            "file:///etc/passwd", "ftp://example.com/", "data:text/html,test",
            "https://user:password@example.com/", "https://@example.com/",
            "https://example.com\\@127.0.0.1/", "https://example.com/\npath",
            "https://example.com:65536/", "https://example.com:bad/",
        ):
            with self.subTest(url=url), patch.object(browser_engines.socket, "getaddrinfo") as dns:
                with self.assertRaises(ValueError):
                    browser_engines.validate_public_url(url)
                dns.assert_not_called()

    def test_nonpublic_literal_addresses(self):
        for host in ("127.0.0.1", "10.1.2.3", "192.168.1.2", "169.254.169.254",
                     "0.0.0.0", "224.0.0.1", "[::1]", "[::ffff:127.0.0.1]"):
            with self.subTest(host=host), self.assertRaises(ValueError):
                browser_engines.validate_public_url(f"http://{host}/")

    def test_dns_must_resolve_exclusively_to_public_addresses(self):
        for addresses in (["10.0.0.1"], ["93.184.216.34", "10.0.0.1"], []):
            with self.subTest(addresses=addresses), patch.object(
                browser_engines.socket, "getaddrinfo", return_value=[dns_record(a) for a in addresses]
            ), self.assertRaises(ValueError):
                browser_engines.validate_public_url("https://example.com/")
        with patch.object(browser_engines.socket, "getaddrinfo", side_effect=socket.gaierror), self.assertRaises(ValueError):
            browser_engines.validate_public_url("https://example.com/")

    def test_internal_fixture_exception_is_loopback_only(self):
        self.assertEqual(
            browser_engines.validate_public_url("http://127.0.0.1:12345/", allow_local_fixture=True),
            "http://127.0.0.1:12345/",
        )
        with patch.object(browser_engines.socket, "getaddrinfo", return_value=[dns_record("127.0.0.1"), dns_record("::1")]):
            self.assertEqual(
                browser_engines.validate_public_url("http://localhost/", allow_local_fixture=True),
                "http://localhost/",
            )
        for host in ("192.168.1.2", "10.0.0.1", "[::1]"):
            with self.subTest(host=host), self.assertRaises(ValueError):
                browser_engines.validate_public_url(f"http://{host}/", allow_local_fixture=True)
        with patch.object(browser_engines.socket, "getaddrinfo", return_value=[dns_record("93.184.216.34")]), self.assertRaises(ValueError):
            browser_engines.validate_public_url("http://localhost/", allow_local_fixture=True)
        with self.assertRaises(ValueError):
            browser_engines.validate_public_url("http://127.0.0.1/", {"example.com"}, allow_local_fixture=True)


class RendererArgumentTests(unittest.TestCase):
    def test_selenium_requires_explicit_driver_setup(self):
        with self.assertRaisesRegex(RuntimeError, "explicit driver_path"):
            browser_engines.render_html(
                "http://127.0.0.1/", "selenium", allow_local_fixture=True
            )

    def test_invalid_engine_timeout_and_selector_do_not_launch(self):
        with patch.object(browser_engines, "_selenium") as selenium, patch.object(browser_engines, "_playwright") as playwright:
            with self.assertRaises(ValueError):
                browser_engines.render_html("https://example.com/", "unknown")
            for timeout in (0, -1, 121, True, "20", float("nan"), float("inf")):
                with self.subTest(timeout=timeout), self.assertRaises(ValueError):
                    browser_engines.render_html("https://example.com/", "playwright", timeout_seconds=timeout)
            for selector in ("", "  ", None):
                with self.subTest(selector=selector), self.assertRaises(ValueError):
                    browser_engines.render_html("https://example.com/", "selenium", wait_selector=selector)
            selenium.assert_not_called()
            playwright.assert_not_called()

    def test_default_render_scope_is_initial_host(self):
        with patch.object(browser_engines.socket, "getaddrinfo", return_value=[dns_record("93.184.216.34")]), patch.object(
            browser_engines, "_playwright", return_value={"engine": "playwright"}
        ) as renderer:
            browser_engines.render_html("https://EXAMPLE.com/", "playwright")
            self.assertEqual(renderer.call_args.args[3], {"example.com"})
            self.assertFalse(renderer.call_args.args[5])

    def test_cleanup_attempts_all_resources_after_one_failure(self):
        warnings = []
        calls = []

        def failing_close():
            calls.append("failing")
            raise OSError("fixture close failure")

        with ExitStack() as resources:
            resources.callback(browser_engines._close, lambda: calls.append("remaining"), warnings)
            resources.callback(browser_engines._close, failing_close, warnings)
        self.assertEqual(calls, ["failing", "remaining"])
        self.assertTrue(warnings)


class FixtureHandler(BaseHTTPRequestHandler):
    def setup(self):
        super().setup()
        # Chromium can preconnect before the routed API fetch sends its GET.
        # Bound idle fixture sockets independently of the render time budget.
        self.connection.settimeout(2)
        accepted = getattr(self.server, "connection_accepted", None)
        if accepted is not None:
            accepted.set()

    def do_GET(self):
        self.server.received_user_agents.append(self.headers.get("User-Agent"))
        body = b"""<!doctype html><html><body><p>early fixture text</p>
        <script>setTimeout(function() {
          const node = document.createElement('p'); node.id = 'ready';
          node.textContent = 'late fixture evidence'; document.body.appendChild(node);
        }, 80);</script></body></html>"""
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_args):
        pass


class FixtureConcurrencyTests(unittest.TestCase):
    def test_idle_preconnection_does_not_block_a_separate_get(self):
        # A single-threaded HTTPServer deadlocks here: it accepts the idle
        # browser connection and never serves the separate route.fetch GET.
        server = ThreadingHTTPServer(("127.0.0.1", 0), FixtureHandler)
        server.received_user_agents = []
        server.connection_accepted = threading.Event()
        thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
        thread.start()
        connection = HTTPConnection("127.0.0.1", server.server_port, timeout=1)
        try:
            with socket.create_connection(("127.0.0.1", server.server_port), timeout=1):
                self.assertTrue(server.connection_accepted.wait(1))
                connection.request("GET", "/", headers={"User-Agent": browser_engines.USER_AGENT})
                response = connection.getresponse()
                self.assertEqual(response.status, 200)
                self.assertIn(b"early fixture text", response.read())
                self.assertIn(browser_engines.USER_AGENT, server.received_user_agents)
        finally:
            connection.close()
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)
            self.assertFalse(thread.is_alive())


@unittest.skipUnless(os.environ.get("TAPA_TEST_BROWSER") in {"selenium", "playwright"}, "Optional browser fixture is not enabled")
class BrowserFixtureTests(unittest.TestCase):
    def test_delayed_dom_content_and_cleanup(self):
        engine = os.environ["TAPA_TEST_BROWSER"]
        driver_path = os.environ.get("TAPA_TEST_DRIVER")
        if engine == "selenium" and not driver_path:
            self.skipTest("Provide an existing TAPA_TEST_DRIVER; fixture does not download drivers")
        profile_paths = []
        owners = []
        original_temporary_directory = browser_engines.tempfile.TemporaryDirectory
        original_close = browser_engines._close

        def temporary_directory(*args, **kwargs):
            directory = original_temporary_directory(*args, **kwargs)
            profile_paths.append(Path(directory.name))
            return directory

        def close(callback, warnings):
            owners.append(getattr(callback, "__self__", None))
            original_close(callback, warnings)

        server = ThreadingHTTPServer(("127.0.0.1", 0), FixtureHandler)
        server.received_user_agents = []
        thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
        thread.start()
        try:
            with patch.object(browser_engines.tempfile, "TemporaryDirectory", side_effect=temporary_directory), patch.object(
                browser_engines, "_close", side_effect=close
            ):
                result = browser_engines.render_html(
                    f"http://127.0.0.1:{server.server_port}/", engine,
                    wait_selector="#ready", timeout_seconds=15, allowed_hosts={"127.0.0.1"},
                    driver_path=driver_path, allow_local_fixture=True,
                )
            self.assertIn('<p id="ready">late fixture evidence</p>', result["html"])
            self.assertEqual(result["engine"], engine)
            self.assertEqual(result["status_code"], 200 if engine == "playwright" else None)
            self.assertIn(browser_engines.USER_AGENT, server.received_user_agents)
            self.assertFalse(any("cleanup reported" in warning for warning in result["warnings"]))
            self.assertGreaterEqual(len(owners), 2)
            self.assertTrue(all(not path.exists() for path in profile_paths))
            for owner in owners:
                if hasattr(owner, "is_connected"):
                    self.assertFalse(owner.is_connected())
                if hasattr(owner, "is_closed"):
                    self.assertTrue(owner.is_closed())
                if hasattr(owner, "process") and owner.process is not None:
                    self.assertIsNotNone(owner.process.poll())
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)
            self.assertFalse(thread.is_alive())


if __name__ == "__main__":
    unittest.main()

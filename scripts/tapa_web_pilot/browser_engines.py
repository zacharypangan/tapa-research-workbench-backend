"""Optional, isolated HTML renderers; caller checks robots/access before calling.

Playwright bounds routed HTTP requests, rejects HTTP redirects, and blocks media,
WebSockets, service workers, and non-GET/HEAD requests. Selenium only guards the
initial/final URL; it cannot enforce equivalent pre-request traffic restrictions.
DNS checks are not IP pinning or an OS-level network sandbox. No user profile,
existing cookies, CAPTCHA handling, proxy, or anti-bot workarounds are used.
"""

from __future__ import annotations

from collections import Counter
from contextlib import ExitStack
import ipaddress
import math
import os
from pathlib import Path
import shutil
import socket
import tempfile
import time
from urllib.parse import urlsplit, urlunsplit


MAX_PAGE_REQUESTS = 100
MAX_RESOURCE_BYTES = 5_000_000
USER_AGENT = "TapaResearchPilot/0.1"
_MEDIA_TYPES = {"image", "media", "font"}


def _host(value: str) -> str:
    return value.rstrip(".").encode("idna").decode("ascii").lower()


def validate_public_url(
    url: str, allowed_hosts: set[str] | None = None, *, allow_local_fixture: bool = False
) -> str:
    """Validate HTTP(S), exact host allowlist, credentials, and resolved addresses.

    The internal fixture exception accepts only localhost/127.0.0.1 resolving
    exclusively to loopback. It is deliberately not a general private-LAN flag.
    """
    if not isinstance(url, str) or "\\" in url or any(ord(c) < 32 or ord(c) == 127 for c in url):
        raise ValueError("URL contains invalid characters")
    parsed = urlsplit(url)
    if parsed.scheme.lower() not in {"http", "https"} or not parsed.hostname:
        raise ValueError("Only absolute HTTP(S) URLs are supported")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("URL credentials are not allowed")
    host = _host(parsed.hostname)
    port = parsed.port  # Also rejects malformed/out-of-range ports.
    if allowed_hosts is not None and host not in {_host(item) for item in allowed_hosts}:
        raise ValueError("URL host is outside allowed_hosts")
    try:
        addresses = {ipaddress.ip_address(host)}
    except ValueError:
        try:
            addresses = {ipaddress.ip_address(row[4][0]) for row in socket.getaddrinfo(
                host, port or (443 if parsed.scheme.lower() == "https" else 80),
                type=socket.SOCK_STREAM,
            )}
        except (OSError, ValueError) as error:
            raise ValueError("URL host could not be resolved safely") from error
    fixture = allow_local_fixture and host in {"localhost", "127.0.0.1"}
    if not addresses or any(
        not address.is_loopback if fixture else (not address.is_global or address.is_multicast)
        for address in addresses
    ):
        raise ValueError("URL resolves to a non-public address")
    authority = f"[{host}]" if ":" in host else host
    if port is not None:
        authority += f":{port}"
    return urlunsplit((parsed.scheme.lower(), authority, parsed.path or "/", parsed.query, parsed.fragment))


def _remaining(deadline: float) -> float:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("Browser render exceeded its time budget")
    return remaining


def _close(callback, warnings: list[str]) -> None:
    try:
        callback()
    except Exception as error:
        warnings.append(f"Resource cleanup reported {type(error).__name__}")


def _executable(value: str) -> str:
    path = Path(value).expanduser().resolve(strict=True)
    if not path.is_file() or not os.access(path, os.X_OK):
        raise ValueError("Browser/driver path must be an executable file")
    return str(path)


def _chrome_binary(value: str | None) -> str:
    if value:
        return _executable(value)
    for candidate in (
        shutil.which("google-chrome"), shutil.which("chromium"),
        shutil.which("chromium-browser"),
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    ):
        if candidate and Path(candidate).is_file():
            return _executable(candidate)
    raise RuntimeError("Specify an already-installed Chrome/Chromium browser_binary")


def _selenium(url, selector, deadline, hosts, binary, driver_path, fixture, warnings):
    if not driver_path:
        raise RuntimeError(
            "Selenium requires an explicit driver_path; provision a compatible "
            "ChromeDriver during approved setup before running collection."
        )
    try:
        from selenium import webdriver
        from selenium.webdriver.chrome.service import Service
        from selenium.webdriver.common.by import By
        from selenium.webdriver.support import expected_conditions as EC
        from selenium.webdriver.support.ui import WebDriverWait
    except ImportError as error:
        raise RuntimeError("Selenium is optional; install it only through the approved setup") from error
    warnings.append(
        "Selenium validates start/final URLs only: redirect hops, subresource hosts/private "
        "addresses, and a per-page request ceiling cannot be enforced by this adapter."
    )
    with ExitStack() as resources:
        profile = resources.enter_context(tempfile.TemporaryDirectory(prefix="tapa-browser-"))
        options = webdriver.ChromeOptions()
        options.binary_location = _chrome_binary(binary)  # Prevent browser auto-download.
        options.page_load_strategy = "eager"
        for argument in (
            "--headless=new", f"--user-data-dir={profile}", "--no-first-run",
            "--no-default-browser-check", "--disable-background-networking",
            "--disable-sync", "--autoplay-policy=user-gesture-required",
            f"--user-agent={USER_AGENT}",
        ):
            options.add_argument(argument)
        options.add_experimental_option("prefs", {
            "profile.managed_default_content_settings.images": 2,
            "profile.default_content_setting_values.cookies": 2,
            "profile.default_content_setting_values.notifications": 2,
            "profile.default_content_setting_values.sound": 2,
            "download_restrictions": 3,
        })
        warnings.append("Selenium disables images/sound/autoplay; it cannot guarantee media downloads are blocked.")
        service = Service(executable_path=_executable(driver_path))
        resources.callback(_close, service.stop, warnings)
        driver = webdriver.Chrome(service=service, options=options)
        resources.callback(_close, driver.quit, warnings)
        driver.set_page_load_timeout(_remaining(deadline))
        driver.set_script_timeout(_remaining(deadline))
        driver.get(url)
        WebDriverWait(driver, _remaining(deadline)).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, selector))
        )
        final_url = validate_public_url(driver.current_url, hosts, allow_local_fixture=fixture)
        return {"html": driver.page_source, "final_url": final_url,
                "status_code": None, "engine": "selenium", "warnings": warnings}


def _playwright(url, selector, deadline, hosts, binary, fixture, warnings):
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as error:
        raise RuntimeError("Playwright is optional; install it only through the approved setup") from error
    blocked = Counter()
    request_count = 0
    with ExitStack() as resources:
        playwright = resources.enter_context(sync_playwright())
        browser = playwright.chromium.launch(
            executable_path=_chrome_binary(binary), headless=True,
            timeout=_remaining(deadline) * 1000,
            args=["--disable-background-networking", "--disable-sync"],
        )
        resources.callback(_close, browser.close, warnings)
        context = browser.new_context(
            accept_downloads=False, service_workers="block", user_agent=USER_AGENT
        )
        resources.callback(_close, context.close, warnings)
        if not hasattr(context, "route_web_socket"):
            raise RuntimeError("Playwright >= 1.48 is required to block WebSockets")
        context.route_web_socket("**/*", lambda websocket: websocket.close())
        page = context.new_page()
        resources.callback(_close, page.close, warnings)
        page.on("popup", lambda popup: popup.close())

        def route_request(route):
            nonlocal request_count
            request_count += 1
            response = None
            try:
                request = route.request
                if request_count > MAX_PAGE_REQUESTS:
                    raise ValueError("request_limit")
                if request.resource_type in _MEDIA_TYPES:
                    raise ValueError("media_disabled")
                if request.method not in {"GET", "HEAD"}:
                    raise ValueError("non_read_method")
                validate_public_url(request.url, hosts, allow_local_fixture=fixture)
                context.clear_cookies()
                headers = {key: value for key, value in request.headers.items()
                           if key.lower() not in {"cookie", "authorization", "proxy-authorization"}}
                response = route.fetch(headers=headers, timeout=_remaining(deadline) * 1000,
                                       max_redirects=0, max_retries=0)
                context.clear_cookies()
                if response.status in {301, 302, 303, 307, 308}:
                    raise ValueError("http_redirect_blocked")
                content = response.body()
                if len(content) > MAX_RESOURCE_BYTES:
                    raise ValueError("resource_size_limit")
                response_headers = {key: value for key, value in response.headers.items()
                                    if key.lower() != "set-cookie"}
                route.fulfill(response=response, body=content, headers=response_headers)
            except Exception as error:
                reason = str(error) if isinstance(error, ValueError) else type(error).__name__
                blocked[reason] += 1
                route.abort("blockedbyclient")
            finally:
                if response is not None:
                    response.dispose()

        context.route("**/*", route_request)
        warnings.extend([
            "HTTP redirects are blocked; pass a prevalidated canonical URL.",
            "Routed requests are limited to 100 per render; media, WebSockets, "
            "service workers, cookies, and non-GET/HEAD requests are disabled.",
            "DNS validation is not IP pinning; browser networking is not an OS-level sandbox.",
        ])
        response = page.goto(url, wait_until="domcontentloaded", timeout=_remaining(deadline) * 1000)
        page.wait_for_selector(selector, state="attached", timeout=_remaining(deadline) * 1000)
        final_url = validate_public_url(page.url, hosts, allow_local_fixture=fixture)
        warnings.extend(f"Blocked {number} request(s): {reason}" for reason, number in sorted(blocked.items()))
        return {"html": page.content(), "final_url": final_url,
                "status_code": response.status if response is not None else None,
                "engine": "playwright", "warnings": warnings}


def render_html(
    url: str, engine: str, *, wait_selector: str = "body", timeout_seconds: float = 20,
    allowed_hosts: set[str] | None = None, browser_binary: str | None = None,
    driver_path: str | None = None, allow_local_fixture: bool = False,
) -> dict:
    """Render once using a fresh browser; no dependency/browser installation.

    Default resource/navigation host scope is the initial host. Explicit hosts
    are exact names, not wildcard suffixes. allow_local_fixture is test-only and
    must not be exposed by a collection CLI. Selenium requires an explicit,
    already-provisioned driver_path; collection never invokes Selenium Manager.
    """
    if engine not in {"selenium", "playwright"}:
        raise ValueError("engine must be selenium or playwright")
    if (isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, (int, float))
            or not math.isfinite(timeout_seconds) or not 0 < timeout_seconds <= 120):
        raise ValueError("timeout_seconds must be finite and in (0, 120]")
    if not isinstance(wait_selector, str) or not wait_selector.strip():
        raise ValueError("wait_selector must be a nonempty CSS selector")
    url = validate_public_url(url, allowed_hosts, allow_local_fixture=allow_local_fixture)
    hosts = allowed_hosts if allowed_hosts is not None else {_host(urlsplit(url).hostname)}
    deadline = time.monotonic() + timeout_seconds
    warnings: list[str] = []
    if engine == "selenium":
        return _selenium(url, wait_selector, deadline, hosts, browser_binary, driver_path,
                         allow_local_fixture, warnings)
    return _playwright(url, wait_selector, deadline, hosts, browser_binary,
                       allow_local_fixture, warnings)

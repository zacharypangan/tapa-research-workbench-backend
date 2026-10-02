"""Bounded crawler for explicitly approved public research hosts."""

import asyncio
import ipaddress
import os
import socket
from urllib.parse import urljoin, urlparse

import httpx

MAX_PAGE_BYTES = 5 * 1024 * 1024

async def checked_url(url: str) -> str:
    parsed = urlparse(url)
    allowed = {host.strip().lower() for host in os.getenv("REPOSITORY_ALLOWED_LINK_HOSTS", "").split(",") if host.strip()}
    host = (parsed.hostname or "").lower()
    if parsed.scheme not in {"http", "https"} or parsed.username or parsed.password or host not in allowed:
        raise ValueError("Source host is not approved for link extraction. Upload the source file instead.")
    if parsed.port not in {None, 80, 443}:
        raise ValueError("Source port is not approved for link extraction.")
    addresses = await asyncio.wait_for(asyncio.to_thread(socket.getaddrinfo, host, parsed.port or (443 if parsed.scheme == "https" else 80), type=socket.SOCK_STREAM), timeout=5)
    for address in addresses:
        ip = ipaddress.ip_address(address[4][0])
        ip = getattr(ip, "ipv4_mapped", None) or ip
        if not ip.is_global:
            raise ValueError("Private network sources cannot be extracted.")
    if not addresses:
        raise ValueError("Source host could not be resolved.")
    return parsed._replace(fragment="").geturl()


async def fetch_html(client: httpx.AsyncClient, url: str) -> httpx.Response:
    async with asyncio.timeout(30):
        return await _fetch_html(client, url)


async def _fetch_html(client: httpx.AsyncClient, url: str) -> httpx.Response:
    for _ in range(6):
        url = await checked_url(url)
        async with client.stream("GET", url) as response:
            if response.is_redirect:
                location = response.headers.get("location")
                if not location:
                    raise ValueError("Source redirect is missing its destination.")
                url = urljoin(url, location)
                continue
            body = bytearray()
            if response.status_code == 200:
                if not response.headers.get("content-type", "").lower().startswith(("text/", "application/xhtml+xml")):
                    raise ValueError("Upload this source file to extract its contents.")
                async for chunk in response.aiter_bytes(chunk_size=64 * 1024):
                    if len(body) + len(chunk) > MAX_PAGE_BYTES:
                        raise ValueError("Source page exceeds the extraction size limit.")
                    body.extend(chunk)
            return httpx.Response(response.status_code, headers=response.headers, content=bytes(body), request=response.request)
    raise ValueError("Source exceeded the redirect limit.")

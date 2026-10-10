"""Bounded PDF downloads from explicitly trusted scholarly hosts."""

import asyncio
import re
from urllib.parse import urljoin, urlsplit

import httpx
import pymupdf

from deepscholar.models import TextPage


def validate_pdf_url(url: str, allowed_hosts: list[str]) -> None:
    parsed = urlsplit(url)
    if (
        parsed.scheme != "https"
        or parsed.hostname not in allowed_hosts
        or parsed.username
        or parsed.password
        or parsed.port not in (None, 443)
    ):
        raise ValueError("PDF host is outside the configured HTTPS allowlist")


async def extract_pdf_text(
    client: httpx.AsyncClient, url: str, max_bytes: int, max_pages: int, allowed_hosts: list[str]
) -> list[TextPage]:
    for _ in range(4):
        validate_pdf_url(url, allowed_hosts)
        async with client.stream("GET", url, follow_redirects=False) as response:
            if response.is_redirect:
                location = response.headers.get("location")
                if not location:
                    raise ValueError("PDF redirect has no destination")
                url = urljoin(url, location)
                continue
            response.raise_for_status()
            length = response.headers.get("content-length")
            if length and int(length) > max_bytes:
                raise ValueError("PDF exceeds configured size limit")
            data = bytearray()
            async for part in response.aiter_bytes():
                data.extend(part)
                if len(data) > max_bytes:
                    raise ValueError("PDF exceeds configured size limit")
            if not data.startswith(b"%PDF"):
                raise ValueError("Downloaded content is not a PDF")
        return await asyncio.to_thread(parse_pdf, bytes(data), max_pages)
    raise ValueError("Too many PDF redirects")


def parse_pdf(data: bytes, max_pages: int) -> list[TextPage]:
    with pymupdf.open(stream=data, filetype="pdf") as document:
        pages = []
        for index in range(min(len(document), max_pages)):
            text = document[index].get_text("text").strip()
            reference_heading = re.search(
                r"(?im)^\s*(?:\d+\.?\s+)?(?:references|bibliography)\s*$", text
            )
            if reference_heading:
                text = text[: reference_heading.start()].strip()
            text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)
            if text:
                pages.append(TextPage(text=text, page=index + 1, kind="pdf"))
            if reference_heading:
                break
        return pages

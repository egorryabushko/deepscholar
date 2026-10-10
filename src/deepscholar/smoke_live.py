"""Explicit, opt-in live network smoke check. Does not run model inference."""

import asyncio
import json
import platform

import httpx

from deepscholar.config import Settings
from deepscholar.connectors.arxiv import ArxivClient
from deepscholar.connectors.openalex import OpenAlexClient
from deepscholar.pdf import extract_pdf_text


async def check() -> dict:
    settings = Settings()
    report = {"python": platform.python_version(), "checks": {}}
    async with httpx.AsyncClient(
        timeout=settings.request_timeout, headers={"User-Agent": "DeepScholar/0.1 smoke check"}
    ) as client:
        sources = {
            "arxiv": ArxivClient(client),
            "openalex": OpenAlexClient(client, settings.openalex_api_key),
        }
        for name, source in sources.items():
            try:
                papers = await source.search("retrieval augmented generation", 2)
                report["checks"][name] = {
                    "status": "passed" if papers else "empty",
                    "papers": len(papers),
                    "titles": [p.title for p in papers],
                    "ids": [p.id for p in papers],
                }
            except Exception as error:
                report["checks"][name] = {"status": "failed", "error": _safe_error(error)}
        try:
            pages = await extract_pdf_text(
                client,
                "https://arxiv.org/pdf/2005.11401",
                settings.max_pdf_bytes,
                2,
                settings.pdf_hosts,
            )
            report["checks"]["pdf"] = {
                "status": "passed" if pages else "empty",
                "pages_with_text": len(pages),
                "characters": sum(len(p.text) for p in pages),
                "url": "https://arxiv.org/pdf/2005.11401",
            }
        except Exception as error:
            report["checks"]["pdf"] = {"status": "failed", "error": _safe_error(error)}
    return report


def _safe_error(error: Exception) -> str:
    if isinstance(error, httpx.HTTPStatusError):
        return f"HTTP {error.response.status_code}"
    return type(error).__name__


if __name__ == "__main__":
    print(json.dumps(asyncio.run(check()), indent=2, ensure_ascii=False))

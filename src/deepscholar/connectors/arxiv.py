import asyncio
import re
import time
import xml.etree.ElementTree as ET

import httpx

from deepscholar.models import Paper

ATOM = {"a": "http://www.w3.org/2005/Atom"}
ARXIV = {"x": "http://arxiv.org/schemas/atom"}


class ArxivClient:
    def __init__(self, client: httpx.AsyncClient, interval: float = 3.0) -> None:
        self.client = client
        self.interval = interval
        self.lock = asyncio.Lock()
        self.last_request = -float("inf")
        self.cache: dict[tuple[str, int], tuple[float, list[Paper]]] = {}

    async def search(self, query: str, limit: int = 5) -> list[Paper]:
        key = (query, limit)
        async with self.lock:
            cached = self.cache.get(key)
            if cached and time.monotonic() - cached[0] < 86400:
                return cached[1]
            delay = self.interval - (time.monotonic() - self.last_request)
            if delay > 0:
                await asyncio.sleep(delay)
            self.last_request = time.monotonic()
            terms = re.findall(r"[\w-]+", query)[:12]
            search_query = " AND ".join(f"all:{term}" for term in terms)
            response = await self.client.get(
                "https://export.arxiv.org/api/query",
                params={
                    "search_query": search_query,
                    "start": 0,
                    "max_results": limit,
                    "sortBy": "relevance",
                },
            )
            response.raise_for_status()
            papers = self.parse(response.text)
            if len(self.cache) >= 256:
                self.cache.pop(next(iter(self.cache)))
            self.cache[key] = (time.monotonic(), papers)
            return papers

    @staticmethod
    def parse(xml: str) -> list[Paper]:
        root = ET.fromstring(xml)
        papers = []
        for entry in root.findall("a:entry", ATOM):
            entry_url = entry.findtext("a:id", default="", namespaces=ATOM).strip()
            if "api/errors" in entry_url:
                raise ValueError("arXiv returned an API error entry")
            if "/abs/" not in entry_url:
                continue
            arxiv_id = entry_url.split("/abs/", 1)[1]
            published = entry.findtext("a:published", default="", namespaces=ATOM)
            papers.append(
                Paper(
                    id=f"arxiv:{arxiv_id}",
                    title=_clean(entry.findtext("a:title", default="", namespaces=ATOM)),
                    abstract=_clean(entry.findtext("a:summary", default="", namespaces=ATOM)),
                    authors=[
                        a.findtext("a:name", default="", namespaces=ATOM)
                        for a in entry.findall("a:author", ATOM)
                    ],
                    year=int(published[:4]) if published[:4].isdigit() else None,
                    source="arxiv",
                    url=f"https://arxiv.org/abs/{arxiv_id}",
                    pdf_url=f"https://arxiv.org/pdf/{arxiv_id}",
                    doi=entry.findtext("x:doi", default=None, namespaces=ARXIV),
                )
            )
        return papers


def _clean(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()

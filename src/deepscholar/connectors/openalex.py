import time

import httpx

from deepscholar.models import Paper


class OpenAlexClient:
    def __init__(self, client: httpx.AsyncClient, api_key: str | None = None) -> None:
        self.client = client
        self.api_key = api_key
        self.cache: dict[tuple[str, int], tuple[float, list[Paper]]] = {}

    async def search(self, query: str, limit: int = 5) -> list[Paper]:
        key = (query, limit)
        cached = self.cache.get(key)
        if cached and time.monotonic() - cached[0] < 86400:
            return cached[1]
        params = {
            "search": query,
            "per_page": limit,
            "select": "id,doi,title,publication_year,authorships,abstract_inverted_index,"
            "primary_location,best_oa_location",
        }
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        response = await self.client.get(
            "https://api.openalex.org/works", params=params, headers=headers
        )
        response.raise_for_status()
        papers = [self._paper(item) for item in response.json().get("results", [])]
        if len(self.cache) >= 256:
            self.cache.pop(next(iter(self.cache)))
        self.cache[key] = (time.monotonic(), papers)
        return papers

    @staticmethod
    def _paper(item: dict) -> Paper:
        location = item.get("best_oa_location") or item.get("primary_location") or {}
        landing = location.get("landing_page_url") or item["id"]
        return Paper(
            id=f"openalex:{item['id'].rsplit('/', 1)[-1]}",
            title=item.get("title") or "Untitled",
            abstract=_restore_abstract(item.get("abstract_inverted_index")),
            authors=[
                a.get("author", {}).get("display_name", "") for a in item.get("authorships", [])
            ],
            year=item.get("publication_year"),
            source="openalex",
            url=landing,
            pdf_url=location.get("pdf_url"),
            doi=item.get("doi"),
        )


def _restore_abstract(index: dict[str, list[int]] | None) -> str:
    if not index:
        return ""
    words = [(position, word) for word, positions in index.items() for position in positions]
    return " ".join(word for _, word in sorted(words))

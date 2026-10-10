from unittest.mock import AsyncMock

import httpx
import pytest

from deepscholar.connectors.arxiv import ArxivClient
from deepscholar.connectors.openalex import OpenAlexClient, _restore_abstract

ARXIV_XML = """<feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
<entry><id>http://arxiv.org/abs/hep-ex/0307015v2</id><title>Title\n here</title>
<summary>Abstract text</summary><published>2003-07-01T00:00:00Z</published>
<author><name>A Author</name></author><arxiv:doi>10.123/example</arxiv:doi></entry></feed>"""

OPENALEX_ITEM = {
    "id": "https://openalex.org/W1",
    "title": "Example title",
    "publication_year": 2024,
    "doi": "https://doi.org/10.123/example",
    "abstract_inverted_index": {"retrieval": [0, 2], "works": [1]},
    "authorships": [{"author": {"display_name": "A Author"}}],
    "best_oa_location": None,
    "primary_location": {
        "landing_page_url": "https://arxiv.org/abs/2005.11401",
        "pdf_url": "https://arxiv.org/pdf/2005.11401",
    },
}


def test_arxiv_metadata_and_legacy_id():
    paper = ArxivClient.parse(ARXIV_XML)[0]
    assert paper.id == "arxiv:hep-ex/0307015v2"
    assert paper.title == "Title here"
    assert paper.pdf_url.path == "/pdf/hep-ex/0307015v2"
    assert paper.year == 2003 and paper.authors == ["A Author"]
    assert paper.doi == "10.123/example"


def test_arxiv_error_entry_is_not_a_paper():
    xml = '<feed xmlns="http://www.w3.org/2005/Atom"><entry><id>http://arxiv.org/api/errors</id></entry></feed>'
    with pytest.raises(ValueError):
        ArxivClient.parse(xml)


async def test_arxiv_query_cache_and_throttle(client_factory, monkeypatch):
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(200, text=ARXIV_XML)

    wait = AsyncMock()
    monkeypatch.setattr("deepscholar.connectors.arxiv.asyncio.sleep", wait)
    async with client_factory(handler) as client:
        source = ArxivClient(client)
        await source.search("retrieval generation", 2)
        await source.search("retrieval generation", 2)
        await source.search("agent reasoning", 2)
    assert len(requests) == 2
    assert requests[0].url.params["search_query"] == "all:retrieval AND all:generation"
    wait.assert_awaited_once()


async def test_openalex_wire_contract_and_cache(client_factory):
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(200, json={"results": [OPENALEX_ITEM]})

    async with client_factory(handler) as client:
        source = OpenAlexClient(client, "test-secret")
        first = await source.search("retrieval")
        assert await source.search("retrieval") == first
    assert len(requests) == 1
    assert requests[0].headers["authorization"] == "Bearer test-secret"
    assert requests[0].url.params["per_page"] == "5"
    assert "test-secret" not in str(requests[0].url)
    assert first[0].abstract == "retrieval works retrieval"
    assert first[0].source == "openalex"


def test_openalex_nullable_fields():
    paper = OpenAlexClient._paper({"id": "https://openalex.org/W2"})
    assert paper.abstract == "" and paper.pdf_url is None
    assert paper.title == "Untitled"
    assert _restore_abstract(None) == ""


async def test_source_http_error_propagates(client_factory):
    async with client_factory(lambda request: httpx.Response(429)) as client:
        with pytest.raises(httpx.HTTPStatusError):
            await OpenAlexClient(client).search("retrieval")

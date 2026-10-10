from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest

from deepscholar.config import Settings
from deepscholar.llm import ExtractiveLLM, HuggingFaceLLM, LocalLLM, build_llm, messages
from deepscholar.models import Chunk, TextPage
from deepscholar.retrieval import chunk_papers, rank_chunks


def test_page_provenance_and_overlap(paper):
    pages = [TextPage(text="retrieval " * 60, page=2, kind="pdf")]
    chunks = chunk_papers([paper], {paper.id: pages}, size=200, overlap=20)
    assert len(chunks) > 1
    assert all(c.page == 2 and c.kind == "pdf" for c in chunks)
    with pytest.raises(ValueError):
        chunk_papers([paper], {}, size=10, overlap=10)


def test_retrieval_filters_zero_overlap_and_ranks(paper):
    a = Chunk(
        paper_id="a",
        paper_title="A",
        text="external retrieval retrieval evidence",
        index=0,
        source_url=paper.url,
    )
    b = a.model_copy(update={"paper_id": "b", "text": "volcanic basalt"})
    assert rank_chunks("retrieval evidence", [b, a])[0].paper_id == "a"
    assert rank_chunks("unrelated galaxies", [a, b]) == []
    assert rank_chunks("the and of", [a]) == []


def test_settings_and_provider_validation():
    with pytest.raises(ValueError):
        Settings(_env_file=None, chunk_overlap=1500)
    with pytest.raises(ValueError):
        Settings(_env_file=None, llm_provider="unknown")
    assert isinstance(build_llm(Settings(_env_file=None)), ExtractiveLLM)
    with pytest.raises(ValueError, match="HF_TOKEN"):
        build_llm(Settings(_env_file=None, llm_provider="huggingface", hf_token=None))
    assert isinstance(build_llm(Settings(_env_file=None, llm_provider="local")), LocalLLM)


async def test_huggingface_adapter_contract(monkeypatch, paper):
    client = AsyncMock()
    client.chat_completion.return_value = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="Supported statement [S1]"))]
    )
    monkeypatch.setattr("deepscholar.llm.AsyncInferenceClient", lambda **kwargs: client)
    model = HuggingFaceLLM(Settings(_env_file=None, hf_token="test", llm_provider="huggingface"))
    chunk = Chunk(
        paper_id=paper.id,
        paper_title=paper.title,
        text=paper.abstract,
        index=0,
        source_url=paper.url,
    )
    answer = await model.synthesize("retrieval evidence", [chunk])
    assert answer.endswith("[S1]")
    assert client.chat_completion.call_args.kwargs["model"] == model.model
    assert "<evidence>" in client.chat_completion.call_args.kwargs["messages"][1]["content"]
    assert "untrusted" in messages("question", [chunk])[0]["content"]
    assert isinstance(
        build_llm(Settings(_env_file=None, hf_token="test", llm_provider="huggingface")),
        HuggingFaceLLM,
    )


async def test_local_adapter_contract(monkeypatch, paper):
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(200, json={"choices": [{"message": {"content": "answer [S1]"}}]})

    real_client = httpx.AsyncClient
    monkeypatch.setattr(
        "deepscholar.llm.httpx.AsyncClient",
        lambda **kwargs: real_client(transport=httpx.MockTransport(handler), **kwargs),
    )
    model = LocalLLM(Settings(_env_file=None, llm_api_key="test-secret"))
    chunk = Chunk(
        paper_id=paper.id,
        paper_title=paper.title,
        text=paper.abstract,
        index=0,
        source_url=paper.url,
    )
    assert await model.synthesize("retrieval evidence", [chunk]) == "answer [S1]"
    assert str(requests[0].url).endswith("/v1/chat/completions")
    assert requests[0].headers["authorization"] == "Bearer test-secret"

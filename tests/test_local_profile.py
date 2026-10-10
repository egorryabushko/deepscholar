import json

import httpx
import pytest
from fastapi.testclient import TestClient

from deepscholar.app import create_app
from deepscholar.config import Settings
from deepscholar.llm import ExtractiveLLM, LocalLLM, messages
from deepscholar.models import Chunk, ResearchRequest
from deepscholar.workflow import ResearchWorkflow


async def test_local_model_and_output_budget(monkeypatch, paper):
    requests = []

    def handler(request):
        requests.append(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"message": {"content": "Evidence [S1]"}}]})

    real_client = httpx.AsyncClient
    monkeypatch.setattr(
        "deepscholar.llm.httpx.AsyncClient",
        lambda **kwargs: real_client(transport=httpx.MockTransport(handler), **kwargs),
    )
    settings = Settings(_env_file=None, llm_model="small-local", max_output_tokens=350)
    evidence = [
        Chunk(
            paper_id=paper.id,
            paper_title=paper.title,
            text=paper.abstract,
            index=0,
            source_url=paper.url,
        )
    ]
    assert await LocalLLM(settings).synthesize("retrieval evidence", evidence) == "Evidence [S1]"
    assert requests[0]["model"] == "small-local"
    assert requests[0]["max_tokens"] == 350
    assert "list: [S1]." in requests[0]["messages"][1]["content"]
    assert "Every factual sentence must end" in messages("question", evidence)[1]["content"]


def test_public_local_profile_has_deadline_but_no_secrets():
    settings = Settings(
        _env_file=None,
        llm_provider="local",
        llm_model="small-local",
        hf_token="private-token",
        workflow_timeout=400,
    )
    with TestClient(create_app(settings)) as client:
        response = client.get("/api/config")
    assert response.json()["model"] == "small-local"
    assert response.json()["workflow_timeout"] == 400
    assert "private-token" not in response.text


async def test_evidence_budget_limits_context(client_factory):
    settings = Settings(_env_file=None, offline=True, evidence_limit=1)
    async with client_factory(lambda request: httpx.Response(404)) as client:
        report = await ResearchWorkflow(settings, client, ExtractiveLLM()).run(
            ResearchRequest(question="retrieval augmented generation")
        )
    assert report.metrics["evidence_chunks"] == 1
    assert len(report.citations) == 1


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_output_tokens": 0},
        {"max_output_tokens": 2049},
        {"evidence_limit": 0},
        {"evidence_limit": 7},
    ],
)
def test_local_budgets_are_bounded(kwargs):
    with pytest.raises(ValueError):
        Settings(_env_file=None, **kwargs)

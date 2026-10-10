import asyncio

import httpx
import pytest
from fastapi.testclient import TestClient

from deepscholar.app import create_app
from deepscholar.config import Settings
from deepscholar.llm import ExtractiveLLM


def test_demo_end_to_end_and_static_ui(settings):
    with TestClient(create_app(settings)) as client:
        assert client.get("/health").json()["status"] == "ok"
        assert client.get("/api/config").json()["mode"] == "demo"
        assert "DeepScholar" in client.get("/").text
        assert client.get("/static/app.js").status_code == 200
        response = client.post(
            "/api/research",
            json={"question": "How does retrieval augmented generation use external evidence?"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["mode"] == "demo"
        assert data["provider"] == "extractive"
        assert data["citations"]
        assert all(c["kind"] == "demo_note" for c in data["citations"])
        assert [step["name"] for step in data["trace"]] == [
            "plan",
            "search",
            "extract",
            "retrieve",
            "synthesize",
        ]
        assert data["metrics"]["pdf_success_rate"] == 0
        assert data["metrics"]["citation_id_validity"] == 1
        assert data["metrics"]["source_coverage"] == 0


@pytest.mark.parametrize(
    "payload",
    [
        {"question": "   "},
        {"question": "abc"},
        {"question": "Valid question", "max_papers": 11},
        {"question": "Valid question", "sources": []},
        {"question": "Valid question", "sources": ["fake"]},
    ],
)
def test_input_validation(settings, payload):
    with TestClient(create_app(settings)) as client:
        assert client.post("/api/research", json=payload).status_code == 422


def test_abstention(settings):
    with TestClient(create_app(settings)) as client:
        data = client.post(
            "/api/research", json={"question": "Volcanic basalt isotope stratigraphy"}
        ).json()
        assert data["citations"] == []
        assert "Insufficient" in data["answer"]


def test_secrets_are_not_exposed():
    config = Settings(_env_file=None, openalex_api_key="secret-value", hf_token="secret-token")
    with TestClient(create_app(config)) as client:
        assert "secret" not in client.get("/api/config").text
        assert "secret" not in repr(config)


def test_server_paper_cap():
    config = Settings(_env_file=None, offline=True, max_papers=1)
    with TestClient(create_app(config)) as client:
        data = client.post(
            "/api/research", json={"question": "Retrieval external generation", "max_papers": 10}
        ).json()
        assert len(data["papers"]) == 1


async def test_busy_and_deadline_release_slot(monkeypatch):
    entered = asyncio.Event()
    release = asyncio.Event()

    async def slow(self, question, evidence):
        entered.set()
        await release.wait()
        return "Answer [S1]"

    monkeypatch.setattr(ExtractiveLLM, "synthesize", slow)
    config = Settings(_env_file=None, offline=True, workflow_timeout=0.15)
    app = create_app(config)
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            payload = {"question": "Retrieval external generation"}
            first = asyncio.create_task(client.post("/api/research", json=payload))
            await entered.wait()
            assert (await client.post("/api/research", json=payload)).status_code == 429
            assert (await first).status_code == 504
            assert not app.state.busy
            release.set()
            assert (await client.post("/api/research", json=payload)).status_code == 200

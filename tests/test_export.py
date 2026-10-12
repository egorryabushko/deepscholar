from fastapi.testclient import TestClient

from deepscholar.app import create_app
from deepscholar.llm import ExtractiveLLM


def test_export_completed_result_without_inference(settings, monkeypatch):
    with TestClient(create_app(settings)) as client:
        result = client.post(
            "/api/research",
            json={"question": "How does retrieval augmented generation use external evidence?"},
        ).json()

        async def forbidden(*args, **kwargs):
            raise AssertionError("Export must not run inference")

        monkeypatch.setattr(ExtractiveLLM, "synthesize", forbidden)
        result["citations"][0].update(kind="pdf", page=7)
        result["citations"][0]["title"] = "Unsafe <script>alert(1)</script> [link](https://x.test)"
        response = client.post("/api/export/markdown", json=result)
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/markdown")
        assert "deepscholar-research.md" in response.headers["content-disposition"]
        report = response.text
        assert result["question"] in report
        assert "[S1]" in report
        assert "PDF excerpt · Page: 7" in report
        assert "<script>" not in report
        assert "&lt;script&gt;" in report
        assert r"\[link\]" in report
        assert result["citations"][0]["url"] in report
        assert "## Warnings" in report
        assert "## Workflow" in report
        assert "## Measurements" in report


def test_export_abstention_and_reject_invalid_payload(settings):
    with TestClient(create_app(settings)) as client:
        result = client.post(
            "/api/research", json={"question": "Volcanic basalt isotope stratigraphy"}
        ).json()
        response = client.post("/api/export/markdown", json=result)
        assert response.status_code == 200
        assert "Insufficient" in response.text
        assert "No supporting evidence was retrieved." in response.text
        assert client.post("/api/export/markdown", json={"answer": "Fake"}).status_code == 422


def test_export_demo_evidence_labels(settings):
    with TestClient(create_app(settings)) as client:
        result = client.post(
            "/api/research", json={"question": "Retrieval external generation"}
        ).json()
        response = client.post("/api/export/markdown", json=result)
        assert "Authored demo note — not a quotation from the paper" in response.text
        assert "Mode: demo" in response.text

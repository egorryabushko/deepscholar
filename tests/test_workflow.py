from unittest.mock import AsyncMock

import httpx

from deepscholar.config import Settings
from deepscholar.llm import ExtractiveLLM
from deepscholar.models import ResearchRequest
from deepscholar.workflow import ResearchWorkflow, _citation_validity, deduplicate


def test_deduplicate_by_title_or_normalized_doi(paper):
    same_title = paper.model_copy(update={"id": "openalex:other", "doi": "10.1/a"})
    same_doi = paper.model_copy(
        update={"id": "arxiv:third", "title": "Other title", "doi": "https://doi.org/10.1/a"}
    )
    assert len(deduplicate([same_title, same_doi, paper])) == 1


def test_citation_validity():
    assert _citation_validity("text [S1] [S3]", 2) == 0.5
    assert _citation_validity("text without a citation", 2) == 0
    assert _citation_validity("text [S0]", 2) == 0


async def test_search_failure_is_reported_without_inventing_sources(client_factory):
    config = Settings(_env_file=None, offline=False)
    async with client_factory(lambda request: httpx.Response(503)) as client:
        workflow = ResearchWorkflow(config, client, ExtractiveLLM())
        workflow.arxiv.interval = 0
        result = await workflow.run(ResearchRequest(question="retrieval external generation"))
    assert not result.papers and not result.citations
    assert result.trace[1].status == "failed"
    assert any("HTTP 503" in warning.message for warning in result.warnings)
    assert "Insufficient" in result.answer


async def test_live_pdf_evidence_and_partial_source_failure(paper, pdf_bytes, client_factory):
    config = Settings(_env_file=None, offline=False)
    async with client_factory(lambda request: httpx.Response(200, content=pdf_bytes)) as client:
        workflow = ResearchWorkflow(config, client, ExtractiveLLM())
        workflow.arxiv.search = AsyncMock(return_value=[paper])
        workflow.openalex.search = AsyncMock(side_effect=RuntimeError("do not disclose token"))
        result = await workflow.run(
            ResearchRequest(question="retrieval generation external documents")
        )
    assert result.mode == "live"
    assert result.metrics["pdf_success_rate"] == 1
    assert result.metrics["source_coverage"] == 0.5
    assert result.citations[0].kind == "pdf"
    assert result.citations[0].page in (1, 2)
    assert len(result.queries) == 2
    assert "do not disclose token" not in result.model_dump_json()


async def test_pdf_failure_uses_labeled_abstract(paper, client_factory):
    config = Settings(_env_file=None, offline=False)
    async with client_factory(lambda request: httpx.Response(404)) as client:
        workflow = ResearchWorkflow(config, client, ExtractiveLLM())
        workflow.arxiv.search = AsyncMock(return_value=[paper])
        result = await workflow.run(
            ResearchRequest(question="retrieval generation", sources=["arxiv"])
        )
    assert result.citations[0].kind == "abstract"
    assert result.metrics["pdf_success_rate"] == 0
    assert result.trace[2].status == "partial"
    assert result.warnings


async def test_bad_model_citations_fall_back(settings, client_factory):
    bad_llm = AsyncMock()
    bad_llm.synthesize.return_value = "Unsupported conclusion [S99]"
    async with client_factory(lambda request: httpx.Response(500)) as client:
        result = await ResearchWorkflow(settings, client, bad_llm).run(
            ResearchRequest(question="retrieval generation external evidence")
        )
    assert result.provider == "extractive"
    assert "[S99]" not in result.answer
    assert any(w.stage == "synthesize" for w in result.warnings)

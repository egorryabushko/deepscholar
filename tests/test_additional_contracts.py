from unittest.mock import AsyncMock

import httpx
import pymupdf

from deepscholar.config import Settings
from deepscholar.evaluate import evaluate
from deepscholar.llm import ExtractiveLLM
from deepscholar.models import ResearchRequest
from deepscholar.pdf import parse_pdf
from deepscholar.workflow import ResearchWorkflow


def test_bibliography_is_not_indexed():
    with pymupdf.open() as document:
        first = document.new_page()
        first.insert_text((72, 72), "Useful evidence about retrieval.")
        second = document.new_page()
        second.insert_text(
            (72, 72), "References\nRetrieval retrieval retrieval irrelevant bibliography."
        )
        third = document.new_page()
        third.insert_text((72, 72), "Continuation of references.")
        pages = parse_pdf(document.tobytes(), 30)
    assert len(pages) == 1
    assert pages[0].page == 1


async def test_requested_sources_share_paper_budget(paper, client_factory):
    second = paper.model_copy(
        update={"id": "openalex:second", "title": "Second independent paper", "source": "openalex"}
    )
    third = paper.model_copy(update={"id": "arxiv:third", "title": "Third paper"})
    async with client_factory(lambda request: httpx.Response(500)) as client:
        workflow = ResearchWorkflow(
            Settings(_env_file=None, offline=False), client, ExtractiveLLM()
        )
        workflow.arxiv.search = AsyncMock(return_value=[paper, third])
        workflow.openalex.search = AsyncMock(return_value=[second])
        papers = await workflow._search("retrieval", ["arxiv", "openalex"], 2, [])
    assert [item.source for item in papers] == ["arxiv", "openalex"]


async def test_extractive_output_does_not_reuse_embedded_citation_ids(
    settings, paper, client_factory
):
    paper = paper.model_copy(update={"abstract": "Retrieval document text with misleading [S99]."})
    async with client_factory(lambda request: httpx.Response(404)) as client:
        workflow = ResearchWorkflow(
            settings.model_copy(update={"offline": False}), client, ExtractiveLLM()
        )
        workflow.arxiv.search = AsyncMock(return_value=[paper])
        response = await workflow.run(
            ResearchRequest(question="Retrieval document", sources=["arxiv"])
        )
    assert "[S99]" not in response.answer
    assert response.metrics["citation_id_validity"] == 1


def test_evaluation_reports_its_narrow_scope():
    report = evaluate()
    assert len(report["cases"]) == 4
    assert report["unrelated_query_abstained"]
    assert "not a general benchmark" in report["scope"]

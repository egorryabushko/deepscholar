import asyncio
import re
import time
from itertools import zip_longest

import httpx
import pymupdf

from deepscholar.config import Settings
from deepscholar.connectors.arxiv import ArxivClient
from deepscholar.connectors.openalex import OpenAlexClient
from deepscholar.demo_data import DEMO_PAPERS
from deepscholar.llm import LLM, ExtractiveLLM
from deepscholar.models import (
    Citation,
    Paper,
    ResearchRequest,
    ResearchResponse,
    TextPage,
    TraceStep,
    WarningMessage,
)
from deepscholar.pdf import extract_pdf_text
from deepscholar.retrieval import chunk_papers, rank_chunks, tokenize


class ResearchWorkflow:
    def __init__(
        self,
        settings: Settings,
        client: httpx.AsyncClient,
        llm: LLM,
        arxiv: ArxivClient | None = None,
        openalex: OpenAlexClient | None = None,
    ) -> None:
        self.settings = settings
        self.client = client
        self.llm = llm
        self.arxiv = arxiv or ArxivClient(client)
        self.openalex = openalex or OpenAlexClient(client, settings.openalex_api_key)

    async def run(self, request: ResearchRequest) -> ResearchResponse:
        trace: list[TraceStep] = []
        warnings: list[WarningMessage] = []
        started = time.perf_counter()
        limit = min(request.max_papers, self.settings.max_papers)
        queries = self._plan(request.question)
        trace.append(_step("plan", started, "ok", f"Primary query: {queries[0]}"))
        search_started = time.perf_counter()
        papers = await self._search(queries[0], request.sources, limit, warnings)
        # Bounded adaptive step: broaden keywords once when primary recall is weak.
        executed_queries = queries[:1]
        if len(papers) < 2 and len(queries) > 1 and not self.settings.offline:
            papers = deduplicate(
                papers + await self._search(queries[1], request.sources, limit, warnings)
            )[:limit]
            executed_queries.append(queries[1])
        status = "failed" if not papers else "partial" if warnings else "ok"
        trace.append(
            _step(
                "search",
                search_started,
                status,
                f"{len(papers)} unique papers; {len(executed_queries)} query rounds",
            )
        )
        if self.settings.offline:
            warnings.append(
                WarningMessage(
                    stage="search",
                    source="demo",
                    message="Fixed corpus with author-written notes. Notes are not paper quotations.",
                )
            )

        fetch_started = time.perf_counter()
        texts, pdf_successes = await self._fetch_texts(papers, warnings)
        trace.append(
            _step(
                "extract",
                fetch_started,
                "ok" if pdf_successes == len(papers) else "partial",
                f"{pdf_successes} PDFs extracted; other evidence explicitly labeled",
            )
        )
        retrieve_started = time.perf_counter()
        chunks = chunk_papers(papers, texts, self.settings.chunk_chars, self.settings.chunk_overlap)
        evidence = rank_chunks(request.question, chunks, limit=6)
        trace.append(
            _step(
                "retrieve",
                retrieve_started,
                "ok" if evidence else "failed",
                f"{len(evidence)} matching BM25 chunks from {len(chunks)} indexed chunks",
            )
        )

        synthesis_started = time.perf_counter()
        provider = self.settings.llm_provider
        synthesis_status = "ok"
        if not evidence:
            provider = "extractive"
            answer = await ExtractiveLLM().synthesize(request.question, [])
            synthesis_status = "partial"
        else:
            try:
                answer = await self.llm.synthesize(request.question, evidence)
                if not answer.strip() or _citation_validity(answer, len(evidence)) < 1:
                    raise ValueError("Response has missing or invalid citation IDs")
            except Exception as error:
                # Never log exception URLs/tokens. Fallback is visible in both trace and response.
                warnings.append(
                    WarningMessage(
                        stage="synthesize",
                        source=provider,
                        message=f"{type(error).__name__}: synthesis unavailable or citations invalid; "
                        "using extractive excerpts.",
                    )
                )
                answer = await ExtractiveLLM().synthesize(request.question, evidence)
                provider = "extractive"
                synthesis_status = "partial"

        # Return only evidence actually referenced in the answer.
        cited_ids = set(int(v) for v in re.findall(r"\[S(\d+)\]", answer))
        citations = [
            Citation(
                id=f"S{i}",
                paper_id=chunk.paper_id,
                title=chunk.paper_title,
                url=chunk.source_url,
                quote=chunk.text[:280],
                page=chunk.page,
                kind=chunk.kind,
            )
            for i, chunk in enumerate(evidence, 1)
            if i in cited_ids
        ]
        validity = _citation_validity(answer, len(evidence))
        trace.append(
            _step(
                "synthesize",
                synthesis_started,
                synthesis_status,
                f"Provider: {provider}; citation ID validity: {validity:.2f}",
            )
        )
        live_sources = {paper.source for paper in papers if paper.source in request.sources}
        return ResearchResponse(
            question=request.question,
            answer=answer,
            citations=citations,
            papers=papers,
            trace=trace,
            mode="demo" if self.settings.offline else "live",
            provider=provider,
            warnings=warnings,
            queries=executed_queries,
            metrics={
                "latency_ms": float(_elapsed(started)),
                "papers_found": float(len(papers)),
                "pdf_success_rate": pdf_successes / len(papers) if papers else 0.0,
                "citation_id_validity": validity,
                "source_coverage": len(live_sources) / len(request.sources),
                "evidence_chunks": float(len(evidence)),
            },
        )

    @staticmethod
    def _plan(question: str) -> list[str]:
        terms = list(dict.fromkeys(tokenize(question)))
        # The MVP planner is deterministic. An LLM planner is a later, evaluated upgrade.
        primary = " ".join(terms[:7]) or question
        broader = " ".join(terms[:3])
        return list(dict.fromkeys([primary, broader])) if broader else [primary]

    async def _search(
        self, query: str, sources: list[str], limit: int, warnings: list[WarningMessage]
    ) -> list[Paper]:
        if self.settings.offline:
            return DEMO_PAPERS[:limit]
        clients = {"arxiv": self.arxiv, "openalex": self.openalex}
        results = await asyncio.gather(
            *(clients[source].search(query, limit) for source in sources), return_exceptions=True
        )
        batches = []
        for source, result in zip(sources, results, strict=True):
            if isinstance(result, BaseException):
                detail = (
                    f"HTTP {result.response.status_code}"
                    if isinstance(result, httpx.HTTPStatusError)
                    else type(result).__name__
                )
                warnings.append(
                    WarningMessage(
                        stage="search",
                        source=source,
                        message=f"{detail}: source search failed; no results fabricated.",
                    )
                )
            else:
                batches.append(result)
        # Interleave source rankings so the first source cannot consume the whole budget.
        papers = [paper for row in zip_longest(*batches) for paper in row if paper is not None]
        return deduplicate(papers)[:limit]

    async def _fetch_texts(
        self, papers: list[Paper], warnings: list[WarningMessage]
    ) -> tuple[dict[str, list[TextPage]], int]:
        if self.settings.offline:
            return {p.id: [TextPage(text=p.abstract, kind="demo_note")] for p in papers}, 0

        async def fetch(paper: Paper) -> tuple[str, list[TextPage], bool]:
            reason = "No PDF URL supplied by source"
            if paper.pdf_url:
                try:
                    pages = await extract_pdf_text(
                        self.client,
                        str(paper.pdf_url),
                        self.settings.max_pdf_bytes,
                        self.settings.max_pdf_pages,
                        self.settings.pdf_hosts,
                    )
                    if pages:
                        return paper.id, pages, True
                    reason = "No extractable PDF text (OCR is not included)"
                except (httpx.HTTPError, ValueError, RuntimeError, pymupdf.FileDataError) as error:
                    reason = (
                        str(error)
                        if isinstance(error, ValueError)
                        else f"{type(error).__name__}: PDF download/extraction failed"
                    )
            warnings.append(
                WarningMessage(
                    stage="extract",
                    source=paper.id,
                    message=f"{reason}; using abstract if available.",
                )
            )
            return paper.id, [TextPage(text=paper.abstract)], False

        # Sequential bounded downloads also avoid bursting requests to arXiv.
        results = []
        for paper in papers:
            results.append(await fetch(paper))
        return {paper_id: text for paper_id, text, _ in results}, sum(ok for _, _, ok in results)


def deduplicate(papers: list[Paper]) -> list[Paper]:
    seen_titles: set[str] = set()
    seen_dois: set[str] = set()
    result = []
    for paper in papers:
        title = re.sub(r"\W+", "", paper.title.casefold())
        doi = re.sub(r"^https?://(dx\.)?doi\.org/", "", paper.doi or "").casefold()
        if title in seen_titles or (doi and doi in seen_dois):
            continue
        seen_titles.add(title)
        if doi:
            seen_dois.add(doi)
        result.append(paper)
    return result


def _elapsed(started: float) -> int:
    return round((time.perf_counter() - started) * 1000)


def _step(name: str, start: float, status: str, detail: str) -> TraceStep:
    return TraceStep(name=name, duration_ms=_elapsed(start), status=status, detail=detail)


def _citation_validity(answer: str, available: int) -> float:
    cited = [int(value) for value in re.findall(r"\[S(\d+)\]", answer)]
    if not cited:
        return 0.0
    return sum(1 <= value <= available for value in cited) / len(cited)

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from deepscholar import __version__
from deepscholar.config import Settings, get_settings
from deepscholar.connectors.arxiv import ArxivClient
from deepscholar.connectors.openalex import OpenAlexClient
from deepscholar.export import render_markdown
from deepscholar.llm import build_llm
from deepscholar.models import ResearchRequest, ResearchResponse
from deepscholar.workflow import ResearchWorkflow

STATIC = Path(__file__).parent / "static"


def create_app(settings: Settings | None = None, transport: httpx.AsyncBaseTransport | None = None):
    config = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Validate provider configuration without running paid model inference.
        build_llm(config)
        async with httpx.AsyncClient(
            timeout=config.request_timeout,
            transport=transport,
            headers={"User-Agent": f"DeepScholar/{__version__} (research portfolio MVP)"},
        ) as client:
            app.state.client = client
            app.state.arxiv = ArxivClient(client)
            app.state.openalex = OpenAlexClient(client, config.openalex_api_key)
            app.state.busy = False
            yield

    app = FastAPI(title="DeepScholar", version=__version__, lifespan=lifespan)
    app.mount("/static", StaticFiles(directory=STATIC), name="static")

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(STATIC / "index.html")

    @app.get("/health")
    def health():
        return {"status": "ok", "version": __version__}

    @app.get("/api/config")
    def public_config():
        selected_model = None
        if config.llm_provider == "huggingface":
            selected_model = config.hf_model
        elif config.llm_provider == "local":
            selected_model = config.llm_model or config.hf_model
        return {
            "mode": "demo" if config.offline else "live",
            "provider": config.llm_provider,
            "max_papers": config.max_papers,
            "model": selected_model,
            "workflow_timeout": config.workflow_timeout,
        }

    @app.post("/api/export/markdown", response_class=Response)
    def export_markdown(result: ResearchResponse):
        return Response(
            render_markdown(result),
            media_type="text/markdown",
            headers={"Content-Disposition": 'attachment; filename="deepscholar-research.md"'},
        )

    @app.post("/api/research", response_model=ResearchResponse)
    async def research(request: ResearchRequest):
        # Check and set happen without awaiting, so admission is atomic in a single worker.
        if app.state.busy:
            raise HTTPException(status_code=429, detail="Research in progress. Try again shortly.")
        app.state.busy = True
        try:
            workflow = ResearchWorkflow(
                config,
                app.state.client,
                build_llm(config),
                app.state.arxiv,
                app.state.openalex,
            )
            async with asyncio.timeout(config.workflow_timeout):
                return await workflow.run(request)
        except TimeoutError:
            raise HTTPException(
                status_code=504, detail="Research deadline exceeded. Narrow the query."
            ) from None
        finally:
            app.state.busy = False

    return app


app = create_app()

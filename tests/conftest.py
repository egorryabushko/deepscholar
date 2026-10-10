import httpx
import pymupdf
import pytest

from deepscholar.config import Settings


@pytest.fixture
def settings():
    return Settings(_env_file=None, offline=True)


@pytest.fixture
def paper():
    from deepscholar.models import Paper

    return Paper(
        id="arxiv:test",
        title="Retrieval augmented generation",
        abstract="Retrieval combines external evidence with generation.",
        source="arxiv",
        url="https://arxiv.org/abs/2005.11401",
        pdf_url="https://arxiv.org/pdf/2005.11401",
    )


@pytest.fixture
def pdf_bytes():
    with pymupdf.open() as document:
        page = document.new_page()
        page.insert_text((72, 72), "Retrieval uses external documents as evidence.")
        page = document.new_page()
        page.insert_text((72, 72), "Page two discusses generation and factual accuracy.")
        return document.tobytes()


@pytest.fixture
def client_factory():
    def factory(handler):
        return httpx.AsyncClient(transport=httpx.MockTransport(handler))

    return factory

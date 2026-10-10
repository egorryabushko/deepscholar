import httpx
import pytest

from deepscholar.pdf import extract_pdf_text, validate_pdf_url


async def test_pdf_pages_and_page_limit(pdf_bytes, client_factory):
    async with client_factory(lambda request: httpx.Response(200, content=pdf_bytes)) as client:
        pages = await extract_pdf_text(
            client, "https://arxiv.org/pdf/test", 100000, 1, ["arxiv.org"]
        )
    assert len(pages) == 1
    assert pages[0].page == 1 and pages[0].kind == "pdf"
    assert "external documents" in pages[0].text


@pytest.mark.parametrize(
    "url",
    [
        "http://arxiv.org/pdf/test",
        "https://localhost/a.pdf",
        "https://127.0.0.1/a.pdf",
        "https://arxiv.org.evil.com/a.pdf",
        "https://arxiv.org:444/a.pdf",
        "https://user:pass@arxiv.org/a.pdf",
    ],
)
def test_unsafe_pdf_url_rejected(url):
    with pytest.raises(ValueError):
        validate_pdf_url(url, ["arxiv.org"])


async def test_redirect_is_rechecked_before_following(client_factory):
    seen = []

    def handler(request):
        seen.append(str(request.url))
        return httpx.Response(302, headers={"location": "http://127.0.0.1/secret"})

    async with client_factory(handler) as client:
        with pytest.raises(ValueError, match="allowlist"):
            await extract_pdf_text(client, "https://arxiv.org/pdf/test", 100000, 2, ["arxiv.org"])
    assert len(seen) == 1


@pytest.mark.parametrize(
    "content,headers,limit,error",
    [
        (b"not a pdf", {}, 1000, "not a PDF"),
        (b"%PDF-oversize", {}, 5, "size limit"),
        (b"%PDF-small", {"content-length": "9999"}, 100, "size limit"),
    ],
)
async def test_bad_or_oversize_download(content, headers, limit, error, client_factory):
    async with client_factory(
        lambda request: httpx.Response(200, content=content, headers=headers)
    ) as client:
        with pytest.raises(ValueError, match=error):
            await extract_pdf_text(client, "https://arxiv.org/pdf/test", limit, 2, ["arxiv.org"])

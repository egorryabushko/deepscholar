"""Small, dependency-free BM25 baseline; scores are relevance, not confidence."""

import math
import re
from collections import Counter

from deepscholar.models import Chunk, Paper, TextPage

TOKEN_RE = re.compile(r"[a-zA-Zа-яА-Я0-9]{2,}")
STOP_WORDS = set(
    "a an the in on of for to and or is are with what how why does do can it its by as this "
    "that from compare between about into research question which evidence improve helps "
    "impact affect".split()
)


def tokenize(text: str) -> list[str]:
    return [t for word in TOKEN_RE.findall(text) if (t := word.lower()) not in STOP_WORDS]


def chunk_papers(
    papers: list[Paper], texts: dict[str, list[TextPage]], size: int, overlap: int
) -> list[Chunk]:
    if not 0 <= overlap < size:
        raise ValueError("Require 0 <= overlap < size")
    chunks = []
    for paper in papers:
        pages = texts.get(paper.id, [TextPage(text=paper.abstract)])
        index = 0
        for page in pages:
            for start in range(0, len(page.text), size - overlap):
                content = page.text[start : start + size].strip()
                if content:
                    chunks.append(
                        Chunk(
                            paper_id=paper.id,
                            paper_title=paper.title,
                            text=content,
                            index=index,
                            source_url=(
                                f"{paper.pdf_url}#page={page.page}"
                                if page.kind == "pdf" and paper.pdf_url and page.page
                                else paper.url
                            ),
                            page=page.page,
                            kind=page.kind,
                        )
                    )
                    index += 1
                if start + size >= len(page.text):
                    break
    return chunks


def rank_chunks(question: str, chunks: list[Chunk], limit: int = 8) -> list[Chunk]:
    query = set(tokenize(question))
    if not chunks or not query:
        return []
    tokens = [Counter(tokenize(chunk.text)) for chunk in chunks]
    df = Counter(token for terms in tokens for token in terms)
    average_length = sum(sum(terms.values()) for terms in tokens) / len(chunks) or 1
    ranked = []
    for chunk, terms in zip(chunks, tokens, strict=True):
        score = 0.0
        length = sum(terms.values())
        for token in query:
            tf = terms[token]
            if tf:
                idf = math.log(1 + (len(chunks) - df[token] + 0.5) / (df[token] + 0.5))
                score += idf * tf * 2.5 / (tf + 1.5 * (0.25 + 0.75 * length / average_length))
        if score > 0:
            ranked.append(chunk.model_copy(update={"score": round(score, 4)}))
    return sorted(ranked, key=lambda item: (-item.score, item.paper_id, item.index))[:limit]

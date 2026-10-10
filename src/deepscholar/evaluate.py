"""Tiny labeled retrieval sanity check; no external services or quality claims."""

import json

from deepscholar.config import Settings
from deepscholar.demo_data import DEMO_PAPERS
from deepscholar.models import TextPage
from deepscholar.retrieval import chunk_papers, rank_chunks

CASES = [
    ("non parametric document index", "demo:rag"),
    ("reasoning actions observations", "demo:react"),
    ("corrective evaluates irrelevant passages", "demo:corrective"),
    ("retrieval generation document provenance", "demo:rag"),
]


def evaluate() -> dict:
    settings = Settings(_env_file=None, offline=True)
    texts = {p.id: [TextPage(text=p.abstract, kind="demo_note")] for p in DEMO_PAPERS}
    chunks = chunk_papers(DEMO_PAPERS, texts, settings.chunk_chars, settings.chunk_overlap)
    rows = []
    for question, target in CASES:
        ranked = rank_chunks(question, chunks, limit=3)
        ids = list(dict.fromkeys(chunk.paper_id for chunk in ranked))
        rank = ids.index(target) + 1 if target in ids else None
        rows.append(
            {
                "question": question,
                "gold_paper": target,
                "retrieved": ids,
                "reciprocal_rank": 1 / rank if rank else 0.0,
            }
        )
    return {
        "scope": "4 lexical sanity cases over 3 authored demo notes; not a general benchmark",
        "recall_at_3": sum(row["reciprocal_rank"] > 0 for row in rows) / len(rows),
        "mrr_at_3": sum(row["reciprocal_rank"] for row in rows) / len(rows),
        "unrelated_query_abstained": not rank_chunks(
            "volcanic basalt isotope stratigraphy", chunks
        ),
        "cases": rows,
    }


if __name__ == "__main__":
    print(json.dumps(evaluate(), ensure_ascii=False, indent=2))

"""Portable reports built from an existing result without repeating inference."""

import re

from deepscholar.models import ResearchResponse


def _text(value: str) -> str:
    """Keep untrusted metadata from introducing HTML or Markdown structure."""
    value = value.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return re.sub(r"([\\`*_{}\[\]#!|])", r"\\\1", value)


def render_markdown(result: ResearchResponse) -> str:
    kinds = {
        "pdf": "PDF excerpt",
        "abstract": "Abstract excerpt",
        "demo_note": "Authored demo note — not a quotation from the paper",
    }
    lines = [
        "# DeepScholar research report",
        "",
        "## Question",
        "",
        _text(result.question),
        "",
        f"Mode: {result.mode} · Provider: {_text(result.provider)}",
        "",
        "## Answer",
        "",
        # Keep citation markers legible while escaping all other result text.
        re.sub(r"\\\[(S\d+)\\\]", r"[\1]", _text(result.answer)),
        "",
        "## Evidence",
        "",
    ]
    if not result.citations:
        lines.extend(["No supporting evidence was retrieved.", ""])
    for cite in result.citations:
        lines.extend(
            [
                f"### {_text(cite.id)} — {_text(cite.title)}",
                "",
                f"Evidence: {kinds[cite.kind]}"
                + (f" · Page: {cite.page}" if cite.page is not None else ""),
                "",
                f"Source: <{cite.url}>",
                "",
                *[f"> {_text(line)}" for line in cite.quote.splitlines()],
                "",
            ]
        )
    lines.extend(["## Selected papers", ""])
    for paper in result.papers:
        lines.extend(
            [
                f"- {_text(paper.title)} ({paper.year or 'year unknown'}; {paper.source})",
                f"  Authors: {_text(', '.join(paper.authors)) or 'Not supplied'}",
                f"  Source: <{paper.url}>",
            ]
        )
    if result.warnings:
        lines.extend(["", "## Warnings", ""])
        lines.extend(
            f"- {_text(w.stage)} / {_text(w.source)}: {_text(w.message)}" for w in result.warnings
        )
    lines.extend(["", "## Search queries", ""])
    lines.extend(f"- {_text(query)}" for query in result.queries)
    lines.extend(["", "## Workflow", ""])
    lines.extend(
        f"- {_text(step.name)}: {step.status}, {step.duration_ms} ms — {_text(step.detail)}"
        for step in result.trace
    )
    lines.extend(["", "## Measurements", ""])
    lines.extend(f"- {_text(name)}: {value:g}" for name, value in result.metrics.items())
    lines.extend(
        [
            "",
            "Citation ID validity checks references only; it does not establish factual support.",
            "This report exports the supplied result and does not rerun or verify the research.",
            "",
        ]
    )
    return "\n".join(lines)

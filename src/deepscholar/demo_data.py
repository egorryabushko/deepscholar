"""Author-written demo notes, NOT verbatim paper abstracts or measured results."""

from deepscholar.models import Paper

DEMO_PAPERS = [
    Paper(
        id="demo:rag",
        title="Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
        abstract=(
            "Retrieval-augmented generation combines parametric memory with a non-parametric "
            "document index. Retrieved passages provide provenance and help factual generation. "
            "A useful research question is how retrieval affects factual question answering."
        ),
        authors=["Patrick Lewis", "Ethan Perez", "Aleksandra Piktus"],
        year=2020,
        source="demo",
        url="https://arxiv.org/abs/2005.11401",
        pdf_url="https://arxiv.org/pdf/2005.11401",
    ),
    Paper(
        id="demo:react",
        title="ReAct: Synergizing Reasoning and Acting in Language Models",
        abstract=(
            "ReAct interleaves reasoning traces with actions that gather external information. "
            "An agent can update its plan from observations. Research questions include how this "
            "workflow changes question answering and interactive decision making."
        ),
        authors=["Shunyu Yao", "Jeffrey Zhao", "Dian Yu"],
        year=2022,
        source="demo",
        url="https://arxiv.org/abs/2210.03629",
        pdf_url="https://arxiv.org/pdf/2210.03629",
    ),
    Paper(
        id="demo:corrective",
        title="Corrective Retrieval Augmented Generation",
        abstract=(
            "Corrective RAG evaluates retrieved documents before generation and triggers alternate "
            "retrieval when evidence quality is low. This reduces the effect of irrelevant passages "
            "and motivates research into robustness and evidence quality."
        ),
        authors=["Shi-Qi Yan", "Jia-Chen Gu"],
        year=2024,
        source="demo",
        url="https://arxiv.org/abs/2401.15884",
        pdf_url="https://arxiv.org/pdf/2401.15884",
    ),
]

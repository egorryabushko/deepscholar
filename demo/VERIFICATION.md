# Verification — first MVP checkpoint

Date: 2026-10-10 (Europe/Moscow). Environment: macOS, Python 3.13.0.

## Completed and observed

- **42 deterministic tests passed** with **92.77% Python statement coverage**.
  Command: `pytest --cov-fail-under=80`.
- `ruff check .` and `ruff format --check .` passed.
- The browser UI loaded and submitted research questions to the running FastAPI backend.
  Demo mode displayed authored notes with explicit labels; an unrelated physics question
  correctly produced an insufficient-evidence response.
- Live search was separately exercised. Both arXiv and OpenAlex returned two papers for
  `retrieval augmented generation`. Raw summary: [live-smoke.json](live-smoke.json).
- Live download/extraction of `https://arxiv.org/pdf/2005.11401` succeeded:
  two pages yielded 7,452 text characters.
- The live browser workflow also returned real physics papers with PDF page numbers.
  Live results depend on network and source availability.
- An offline four-case retrieval sanity check passed; results:
  [retrieval-evaluation.json](retrieval-evaluation.json).
  This is a tiny development fixture, **not evidence of general research quality**.
- Wheel and source distribution builds succeeded. Wheel inspection confirmed that
  the HTML, CSS and JavaScript UI assets are included.
- The locked development dependencies were accepted by a pip dry run. This was run
  against the installed environment, not a clean Linux container.
- Published baseline `d5f085c` passed all three GitHub Actions jobs on clean Ubuntu runners:
  tests on Python **3.11 and 3.13**, and Docker build plus runtime health/research smoke check.
  Evidence: [CI run #1](https://github.com/egorryabushko/deepscholar/actions/runs/38047963343).
  Later commits add stronger container response/static-asset assertions; check their own runs.

## Implemented but not live-verified here

- Hugging Face chat provider and local OpenAI-compatible open-weight model adapter:
  tested using mocked responses. No HF token or running local model server was provided.
  No model inference or provider billing occurred.
- **Docker was unavailable on this computer; container build/runtime verification ran in CI,
  not locally.** Compose's read-only/capability restrictions have not been runtime-tested here.
- The public GitHub repository `egorryabushko/deepscholar` was created after local verification.
  Check the current
  [Actions runs](https://github.com/egorryabushko/deepscholar/actions) for the published commit.

## Limits at this checkpoint

The running preview uses live search with deterministic extractive output.
The repository's default launch mode remains the reproducible offline demo.
The current system retrieves excerpts; synthesized LLM research answers require
a configured provider. Citation ID validity does not validate factual entailment.

BM25 uses lexical overlap and can return mathematically dense passages.
Indexing stops at an explicit References/Bibliography heading to reduce bibliography matches;
appendices after that heading are also omitted. Up to 30 PDF pages are indexed.
This MVP does not include OCR, embeddings, a persistent vector store, authentication
or durable background jobs.

The test run emitted dependency deprecation warnings from PyMuPDF/SWIG and Starlette's
httpx-based test client; they did not cause test failures.

## Next checkpoint

1. Configure an available open-weight model, verify a real answer end to end,
   and evaluate claim support on an independently labeled question set.
2. Add semantic/hybrid retrieval only after comparing it with this baseline.

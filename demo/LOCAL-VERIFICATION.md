# Local model verification — no paid inference API

Date: 2026-10-10. macOS 15.8.1, Intel Core i5-8257U, 8 GiB RAM.

## Observed

- Downloaded official Qwen2.5-1.5B-Instruct GGUF Q4_K_M weights, 1,117,320,736 bytes.
  Revision: `91cad51170dc346986eccefdc2dd33a9da36ead9`.
  SHA-256 matched HF LFS metadata:
  `6a1a2eb6d15622bf3c96857206351ba97e1af16c30d7a74ee38970e434e9407e`.
- Used the official llama.cpp macOS x64 release `b11429`, CPU-only, 4 threads,
  a 4,096-token context and one slot. The archive SHA-256 matched release metadata:
  `29ac3ea02be6bd143e824973f2cc5fa74bc4094393a9eaab0ff6814f19dd8522`.
- Model and runtime stayed outside the Git repository; no system-wide runtime installation.
  The server binds to `127.0.0.1:8080`, with a restricted CORS origin and no agent tools.
- A live end-to-end API request returned `provider: local`, not an extractive fallback:
  3 articles, 3 extracted PDFs, 3 selected evidence chunks, citations S1–S3.
- The recorded run took **9,263 ms** with a warm model/prompt cache. A prior cold prompt
  took about 26.6 seconds for model processing alone. These are observed cases, not a
  latency guarantee. Raw reduced report: [local-smoke.json](local-smoke.json).
- A separate browser smoke run finished in **15.31 seconds**, displayed `LIVE · LLM`
  and `Provider: local`, and had no synthesis fallback warning.
  Screenshot: [local-screenshot.jpg](local-screenshot.jpg).
- **49 deterministic tests passed**, Python statement coverage **92.73%**;
  Ruff checks and formatting passed. Tests use mocked inference except this explicit smoke case.
- The implementation commit `df617b4` passed Python 3.11/3.13 tests and Docker smoke checks
  in [CI run](https://github.com/egorryabushko/deepscholar/actions/runs/38049241288).
  CI does not download model weights or test model quality.

## Failure paths and limits

Initial free-text responses omitted source IDs; the workflow rejected them and displayed
an explicit extractive fallback. The prompt now enumerates exact allowed IDs and specifies
their formatting. A subsequent live question passed the ID check. This is not a guarantee
that every question will produce a compliant answer.

The recorded answer groups citations at the end of a paragraph. Citation ID validity is
**not claim-level support, entailment, or correctness**. This small model and lexical retrieval
still need independently labeled evaluation; there is no answer-quality score here.
Weak or unrelated evidence can produce poor results or fallbacks. No automatic source
IDs are appended to uncited model output.

This mode does not call paid inference APIs or send generation prompts to a cloud LLM.
It still uses local CPU, memory and electricity. Live scholarly search sends keyword
queries to arXiv/OpenAlex. The previously configured HF token remains locally ignored,
unused by the local adapter, and is not included in artifacts or the repository.

## Restart

Run the model server with `scripts/serve-local.sh` and a compatible llama-server executable
and GGUF path, then start FastAPI with the settings in `profiles/local-cpu.env.example`.
See the README's local CPU section. The project does not start a system login service.

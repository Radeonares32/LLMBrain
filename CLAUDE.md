# CLAUDE.md

## Project Overview

LLM Brain is a source-grounded engineering memory compiler. It reads repositories,
documentation, incidents, ADRs, and security notes, then compiles durable memory
artifacts: SQLite/JSONL storage, BrainFrame compact LLM context, Markdown/MDX wiki
pages, and graph knowledge maps.

Slogan: **RAG answers questions. LLM Brain builds memory.**

## Core Architecture

- SQLite + JSONL are canonical storage.
- BrainFrame is the token-efficient LLM input context.
- JSON Schema is used for structured LLM output.
- Markdown/MDX are human-readable docs.
- `graph.json` and `graph.graphml` are graph exports.
- CLI-first workflows are primary.
- FastAPI is an optional server surface.
- The project is open-source-first and CI/CD-ready.

## Non-Negotiable Rules

- Do not replace BrainFrame LLM context with large JSON blobs.
- Do not remove source evidence from facts, relations, or wiki pages.
- Do not send secrets to LLM providers.
- Do not make mock/offline test mode require API keys.
- Do not break CLI-first workflows.
- Do not couple storage directly to a specific LLM provider.
- Do not introduce hidden network calls in tests.
- Do not modify user source files during scanning/building.

## Development Commands

```bash
pip install -e ".[dev,providers,web]"
python -m compileall llmbrain tests
pytest
llmbrain --help
llmbrain version
llmbrain build examples/sample-project --provider deepseek
llmbrain token-report examples/sample-project
llmbrain ci examples/sample-project --fail-on high
```

## Package Layout

- `llmbrain/cli.py`: Typer/Rich command-line interface.
- `llmbrain/main.py`: FastAPI application entrypoint.
- `llmbrain/core/`: settings and application configuration.
- `llmbrain/api/`: optional HTTP API routes.
- `llmbrain/models/`: Pydantic data models.
- `llmbrain/services/`: scanner, chunker, extraction, wiki, graph, benchmark, orchestration.
- `llmbrain/storage/`: SQLite, JSONL, and filesystem artifact writers.
- `llmbrain/llm/`: provider abstractions and production provider implementations.
- `llmbrain/formats/`: BrainFrame and other output format helpers.
- `docs/`, `examples/`, `tests/`: documentation, fixtures, and verification.

## Important Modules

- `llmbrain/formats/brainframe.py`: canonical BrainFrame renderer and escaping rules.
- `llmbrain/services/context_builder.py`: compatibility wrapper for BrainFrame context.
- `llmbrain/services/token_budget.py`: JSON-vs-BrainFrame token comparison helpers.
- `llmbrain/services/redactor.py`: secret redaction before LLM input and artifacts.
- `llmbrain/services/evidence_health.py`: evidence coverage scoring.
- `llmbrain/llm/providers.py`: OpenAI-compatible, Anthropic, and Ollama provider adapters.
- `llmbrain/services/project_service.py`: full build orchestration and incremental reuse.

## LLM Provider Rules

- Production providers are optional and configured through environment variables.
- OpenAI and DeepSeek are supported production providers.
- DeepSeek should be treated as OpenAI-compatible where possible.
- Anthropic and Ollama should remain provider-isolated.
- Mock/offline test providers must always work without API keys.
- Missing API keys must not break import, package metadata, CLI help, or tests.
- Offline tests should use fakes/stubs and must not call real provider APIs.

## Token Efficiency Rules

- LLM input context must use BrainFrame.
- JSON is okay for APIs, storage, manifest, graph, JSONL, and JSON Schema output validation.
- JSON is not okay as the default large LLM prompt context.
- Use `llmbrain token-report PATH` to compare JSON-style context vs BrainFrame.

## Evidence Rules

- Every fact should have source evidence.
- Evidence format: `path:Lstart-Lend`.
- Invalid evidence cannot be high confidence.
- Wiki pages should include source evidence.
- Hallucination guard and evidence health checks must remain active.

## Security Rules

- Redact secrets before LLM context.
- Do not mutate original files.
- Preserve line numbers after redaction.
- Treat `.env`, tokens, private keys, passwords, and API keys carefully.

## Testing Rules

- Tests must not call real LLM APIs.
- Tests must use mock providers, fakes/stubs, or local fixtures.
- Tests should use temp directories where possible.
- CLI smoke tests should verify help/version/build behavior.

## Contribution Guidance for Agents

- Prefer small, focused changes.
- Keep public APIs stable.
- Update README/docs when CLI behavior changes.
- Add tests for behavior changes.
- Run verification commands before claiming completion.

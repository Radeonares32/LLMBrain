# LLM Brain

> RAG answers questions. **LLM Brain builds memory.**

Source-grounded engineering memory compiler for repositories, docs, incidents, ADRs, and security notes.

Traditional RAG retrieves relevant chunks for a question. LLM Brain compiles repositories and documentation into durable, auditable engineering memory: SQLite/JSONL storage, compact BrainFrame context, Markdown/MDX wiki pages, and graph knowledge maps.

[![CI](https://img.shields.io/github/actions/workflow/status/Radeonares32/LLMBrain/ci.yml?branch=main&label=ci)](https://github.com/Radeonares32/LLMBrain/actions/workflows/ci.yml)
[![Version](https://img.shields.io/badge/version-0.1.0-blue.svg)](pyproject.toml)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue.svg)](pyproject.toml)

## Why LLM Brain?

*RAG is useful for retrieval. But engineering teams also need persistent memory.*

Turn any repository into source-grounded engineering memory. While standard Retrieval-Augmented Generation (RAG) tools fetch snippets to answer one-off questions, LLM Brain turns scattered knowledge into versioned, source-grounded artifacts.

That is why the project's philosophy is: **RAG answers questions. LLM Brain builds memory.**

### What it is
- A memory compiler for engineering teams.
- An automated pipeline that extracts facts, entities, and relations from code and docs.
- A knowledge graph generator.
- A secure redactor that protects your secrets before hitting an LLM.

### What it is not
- It is **not** a traditional RAG chatbot.
- It is **not** an AI auto-completion tool.
- It is **not** a generic documentation generator that hallucinates without evidence.

## Approach Comparison

| Approach          | Primary job                | Output                                 | Limitation                                       |
| ----------------- | -------------------------- | -------------------------------------- | ------------------------------------------------ |
| Traditional RAG   | Retrieve chunks            | Temporary context                      | Does not create durable memory                   |
| AI docs generator | Generate documentation     | Markdown/docs                          | Often lacks evidence and reproducibility         |
| **LLM Brain**     | Compile engineering memory | SQLite, JSONL, BrainFrame, Wiki, Graph | Designed for source-grounded knowledge workflows |

## Architecture

From codebase chaos to auditable knowledge, LLM Brain employs a structured pipeline:
1. **Scanner & Chunker**: Safely parses `.md`, `.py`, `.ts`, `Dockerfile`, etc., ignoring binaries and secrets.
2. **LLM Extraction**: Integrates via `openai`, `deepseek`, `anthropic`, or `ollama` to extract structured data via strict JSON Schemas.
3. **Security & Evidence Layer**: Automatically strips secrets and verifies that extracted facts actually exist in the cited source lines. Hallucinations are demoted in confidence.
4. **Storage Engine**: Persists data canonically in SQLite and exports as JSONL.
5. **Generators**: Compiles BrainFrame (`.bf`) token-efficient context, a markdown Wiki, and GraphML maps.

## Features

- **Evidence-first memory for software teams**: Every fact extracted is tied to a specific `path:Lstart-Lend`.
- **Secret Redaction**: Safely strips out API keys and tokens (e.g. AWS, OpenAI) before they leave your machine.
- **Hallucination Guard**: Automatically detects facts lacking source evidence and flags them.
- **CI/CD Ready**: Native GitHub Actions support to detect documentation drift and score PR risks.
- **Provider Agnostic**: Use Local LLMs or Cloud Providers seamlessly.

## Installation

Install from GitHub:

```bash
git clone https://github.com/Radeonares32/LLMBrain.git
cd LLMBrain
pip install -e ".[dev,providers,web]"
```

Or install directly from the repository:

```bash
pip install "llmbrain @ git+https://github.com/Radeonares32/LLMBrain.git"
```

PyPI package installation is coming after the first public release.

## Quickstart

Clone the project and run the CLI:

```bash
git clone https://github.com/Radeonares32/LLMBrain.git
cd LLMBrain
pip install -e .
llmbrain --help
```

Build an auditable engineering memory for the sample project using a production
LLM provider:

```bash
export DEEPSEEK_API_KEY=...
llmbrain build examples/sample-project --provider deepseek
llmbrain token-report examples/sample-project
```

Print the token-compact BrainFrame context:
```bash
llmbrain context examples/sample-project --print
```

Generate a knowledge graph:
```bash
llmbrain graph examples/sample-project --format json
```

Run CI checks on a project to detect drift and hallucinated evidence:
```bash
llmbrain ci examples/sample-project --fail-on high
```

## Interactive Terminal User Interface (TUI)

You can launch a fully interactive terminal user interface directly inside any repository by running:

```bash
llmbrain
```

Or target a specific project directory:

```bash
llmbrain /path/to/project
```

The TUI provides a responsive multi-pane dashboard where you can:
- Ask architecture questions or run coding tasks in chat pane.
- Interactively approve or deny sensitive operations (`[A] Approve once`, `[S] Approve session`, `[D] Deny`).
- Inspect workspace Git diffs and run unit tests.
- View and switch between multiple durable sessions using keyboard shortcuts (Ctrl+X leader commands).
- Monitor async indexing queue status and CPU/memory resource utilization in the live **Observe Panel** (focus via `Ctrl+X O`).

---

## CLI Usage

LLM Brain is a CLI-first tool. Its commands are organized into core compilation commands, agent loops, and utility namespaces.

### Core Compilation Commands
- `llmbrain scan PATH`: Scans a directory and reports supported files, ignored files, and hashes.
- `llmbrain build PATH`: Runs the compilation pipeline (cold build).
- `llmbrain index PATH`: Builds and refreshes project index database incrementally (uses cached elements based on file hashes).
- `llmbrain context PATH`: Outputs the generated token-compact BrainFrame (`.bf`) file.
- `llmbrain diff PATH`: Shows git diffs and tracks memory drift.
- `llmbrain health PATH`: Generates an evidence health score for the repository.
- `llmbrain token-report PATH`: Compares JSON-style context size with BrainFrame context size.
- `llmbrain search "<query>"`: Performs semantic search over the compiled project memory (chunks, facts, entities) using local embeddings.
- `llmbrain doctor`: Runs diagnostic health checks on databases and the environment.
- `llmbrain config`: Prints active configuration settings (API providers, models, local directories).
- `llmbrain logs`: Prints application logs location.
- `llmbrain serve`: Boots the FastAPI server for programmatic access.

### Specialized Agent Loops
Execute single-turn commands or launch interactive coding loops against the repository memory:
- `llmbrain run "<task>"`: Executes a task with automatic routing to the most suitable specialized agent.
- `llmbrain ask "<question>"`: Q&A about the repository architecture, symbols, and decisions (runs `ask` agent in `read-only` mode).
- `llmbrain plan "<task>"`: Generates a detailed step-by-step implementation plan for a coding task.
- `llmbrain build "<task>"`: Automatically implements, tests, and verifies a coding task (runs `build` agent in `ask-before-write` mode).
- `llmbrain review`: Performs code correctness and security reviews on uncommitted changes.
- `llmbrain debug "<problem>"`: Reproduces and diagnoses bugs.
- `llmbrain test "<task>"`: Inspects coverage, writes new tests, or runs test suites.
- `llmbrain security "<scope>"`: Performs a secure code review over the specified scope.

### Multi-Repository Registry Management (`llmbrain repo`)
Manage multiple local codebases inside a centralized registry:
- `llmbrain repo add PATH`: Register a local codebase root directory into the registry.
- `llmbrain repo remove PROJECT_ID`: Remove a codebase registry entry.
- `llmbrain repo list`: List all registered project paths, IDs, and tags.
- `llmbrain repo tag PROJECT_ID TAG`: Add a classification tag to a registered repository.

### Live Observability (`llmbrain observe`)
Inspect background services, resource allocations, and job processing:
- `llmbrain observe queue-stats`: Display indexing job queue statistics.
- `llmbrain observe profiler-report`: Print operation profiler execution times and memory deltas.
- `llmbrain observe resource-status`: Print real-time system CPU & memory resource metrics.
- `llmbrain observe services`: Print health check states of remote LLM providers and server endpoints.

### Agent Session Management (`llmbrain sessions`)
Manage saved agent conversation sessions and TUI states:
- `llmbrain sessions list`: List all active and archived sessions.
- `llmbrain sessions new`: Create a new session.
- `llmbrain sessions resume SESSION_ID`: Resume a session and start the TUI.
- `llmbrain sessions rename SESSION_ID NEW_TITLE`: Rename a session.
- `llmbrain sessions archive SESSION_ID`: Mark session status as archived.
- `llmbrain sessions delete SESSION_ID`: Delete a session.
- `llmbrain sessions export SESSION_ID`: Export conversation transcript in markdown.

### Specialized Agents Metadata (`llmbrain agents`)
Manage and validate specialized agent configurations:
- `llmbrain agents list`: List all available specialized agents.
- `llmbrain agents show AGENT_NAME`: Show detailed configuration (context budget, tools, safety mode) for an agent.
- `llmbrain agents validate`: Validate configuration schemas and permissions for all agents.

### Repository Memory & Database Management (`llmbrain db` / `llmbrain memory`)
- `llmbrain memory inspect`: Review task history, architectural decisions made by the agent, command log, and failures.
- `llmbrain memory refresh`: Force rebuild of the repository memory database.
- `llmbrain db backup --output backup.zip`: Create a zip backup of project database files.
- `llmbrain db restore backup.zip`: Restore project databases from a zip backup.

### Cache Management (`llmbrain cache`)
- `llmbrain cache stats`: Display hit/miss metrics and memory cache size.
- `llmbrain cache clear`: Clear global cache.
- `llmbrain cache clear --project`: Clear cache only for the current project.

Run `llmbrain --help` for a full list of commands and options.


## Incremental Builds

Cold builds can be expensive because source-grounded fact/entity extraction requires
many structured LLM calls. LLM Brain now supports incremental reuse by default:

- unchanged chunk facts are reused from previous JSONL artifacts,
- unchanged document entities are reused,
- no-change relation graphs are reused,
- changed files still flow through the normal provider + JSON Schema validation path.

Use `--full` to force a complete rebuild:

```bash
llmbrain build . --provider deepseek --full
```

## Benchmark Snapshot

The latest full-repository benchmark was run with **DeepSeek**. These numbers
should be read as DeepSeek-validated results; OpenAI, Anthropic, and Ollama may
work through their provider adapters, but this benchmark has not been verified
against those providers yet.

The result shows the intended split: BrainFrame is the compact LLM context, while
Markdown wiki and JSONL remain durable memory artifacts rather than prompt
payloads.

```mermaid
xychart-beta
    title "Estimated Tokens by Context Format"
    x-axis ["JSON context", "BrainFrame", "Retrieval top5"]
    y-axis "Estimated tokens" 0 --> 380000
    bar [370501, 29991, 126]
```

| Metric | Value |
| --- | ---: |
| Validated provider | DeepSeek |
| Other providers | Not benchmarked yet |
| Source estimate | 85,108 tokens |
| JSON-style context baseline | 370,501 tokens |
| BrainFrame context | 29,991 tokens |
| BrainFrame savings vs JSON | 91.91% |
| BrainFrame savings vs source | 64.78% |
| Retrieval top-5 context | 126 tokens |
| Retrieval top-5 savings vs source | 99.85% |
| No-change incremental build | 8.06s without API key |
| Evidence health | 100/100 |

## Agent-Friendly Development

LLM Brain includes agent-oriented project instructions:

- `CLAUDE.md` — coding-agent guidance for Claude/Cursor-style tools
- `AGENTS.md` — recommended agent roles and responsibilities
- `SKILLS.md` — modular capability map for LLM Brain skills

These files help AI coding agents preserve the project architecture,
evidence-first behavior, token-efficient BrainFrame context, and
mock/offline-compatible workflows.

## FastAPI Usage

LLM Brain can also be used programmatically or run as an API server:

```bash
llmbrain serve --host 0.0.0.0 --port 8000
```

Once running, the server exposes the following observability and search endpoints:
- **`GET /observe/queue`**: Retrieve current indexing job queue statistics for a project path.
- **`GET /observe/queue/{project_id}/jobs`**: Fetch pending/running/completed job details.
- **`GET /observe/resource`**: Get CPU, memory metrics, and worker concurrency recommendations.
- **`GET /observe/health`**: Automated health monitor for remote dependencies/LLM endpoints.
- **`GET /observe/profiler`**: Profile report listing duration and virtual memory deltas of operations.
- **`GET /observe/semantic-search`**: Query text and return semantic matching facts, chunks, and entities.

Or via the Python API:

```python
from llmbrain.services.project_service import ProjectService

service = ProjectService()
result = service.build_project("examples/sample-project")
print(f"Compiled {result.document_count} documents.")
```

## Output Structure

Compiling a project produces source-grounded knowledge maps for modern codebases under the `.llmbrain/` directory:

```text
.llmbrain/
├── brain.db                 # Canonical SQLite storage
├── manifest.json            # Build run metadata
├── documents.jsonl          # Raw scanned documents
├── chunks.jsonl             # Extracted code chunks
├── facts.jsonl              # Extracted engineering facts
├── entities.jsonl           # Extracted entities
├── relations.jsonl          # Extracted relations
├── llm-context/
│   └── brainframe.bf        # Token-compact context for LLMs
├── schemas/                 # Strict JSON schemas used for extraction
├── graph/
│   ├── graph.json           # Knowledge Graph representation
│   └── graph.graphml        # GraphML format for visualization
└── wiki/                    # Generated Markdown Wiki
    ├── index.md
    └── project-overview.md
```

## Token-Efficient LLM Context

LLM Brain stores canonical data in SQLite and JSONL, but it does not send large JSON blobs to LLMs by default. For LLM context, it uses BrainFrame: a compact TOON/JTON-style table format designed to reduce repeated keys and preserve source evidence.

Use `llmbrain token-report PATH` to compare a JSON-style context baseline with the generated BrainFrame context.

## BrainFrame Example

`brainframe.bf` is our ultra-compact, token-efficient format designed specifically to feed LLMs maximum context with minimum overhead:

```text
@project sample-project
@project_id 1234abcd
@type engineering_knowledge_compiler

#entities
auth_service | service | AuthService | app/services/auth.py | high

#relations
main.py | uses | auth_service | app/main.py:L10-L12 | high

#facts
f1 | AuthService | handles | user login | app/services/auth.py:L5-L15 | high
```

## Security & Evidence Model

LLM Brain prioritizes security and truthfulness.
- **Redaction**: Secrets like `OPENAI_API_KEY=sk-...` are regex-matched and replaced with `[REDACTED_API_KEY]` entirely locally.
- **Evidence Verification**: All facts must point to a valid line range in a real file. Invalid pointers cause the fact's confidence score to drop to `weak` or `invalid`.

## CI/CD & GitHub Actions

You can integrate LLM Brain directly into your CI/CD pipelines to monitor drift between your code and your documentation. The `llmbrain ci` command will generate a `ci-result.json` and fail the build if high-risk regressions are detected.

Example GitHub Action (`.github/workflows/llmbrain-ci.yml`):
```yaml
name: LLM Brain CI
on: [pull_request]
jobs:
  check-drift:
    runs-on: ubuntu-latest
    env:
      DEEPSEEK_API_KEY: ${{ secrets.DEEPSEEK_API_KEY }}
    steps:
      - uses: actions/checkout@v6
      - run: pip install "llmbrain @ git+https://github.com/Radeonares32/LLMBrain.git"
      - run: llmbrain ci . --provider deepseek --fail-on high
```

## LLM Providers

LLM Brain supports multiple providers via the `--provider` flag:
- `openai`: Requires `OPENAI_API_KEY`.
- `deepseek`: Requires `DEEPSEEK_API_KEY`.
- `anthropic`: Requires `ANTHROPIC_API_KEY`.
- `ollama`: Requires `OLLAMA_MODEL` and a running Ollama server.

## Roadmap
See [ROADMAP.md](ROADMAP.md) for planned features, including full SaaS integration and advanced Web UIs.

## Contributing
We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) and our [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).

## License
MIT License. See [LICENSE](LICENSE) for details.

---

If your team needs more than chat over files, LLM Brain is built for that.

**RAG answers questions. LLM Brain builds memory.**

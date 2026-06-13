# AGENTS.md

## Purpose

This file describes recommended agent roles for agentic development on LLM Brain.
It helps Claude, Cursor, Codex, Copilot, and future coding agents preserve the
project architecture, evidence-first behavior, token-efficient BrainFrame context,
and CLI-first workflows.

## Agent Roles

### 1. Architecture Agent

Responsibilities:

- Preserve architectural consistency.
- Maintain the SQLite/JSONL/BrainFrame/JSON Schema separation.
- Review whether new features fit the core design.

### 2. Scanner Agent

Responsibilities:

- File scanning.
- Skip rules.
- Binary and non-UTF8 handling.
- `content_hash`.
- `line_count`.
- Symlink safety.

### 3. Memory Storage Agent

Responsibilities:

- SQLite schema.
- JSONL export.
- Manifest files.
- Build runs.
- Project registry resolution.

### 4. BrainFrame Agent

Responsibilities:

- Token-efficient context.
- TOON/JTON-style tables.
- Pipe escaping.
- Truncation.
- `token-report`.

### 5. LLM Provider Agent

Responsibilities:

- Provider interface.
- OpenAI, DeepSeek, Ollama, and Anthropic adapters.
- Mock/offline fakes and stubs for tests.
- Structured output.
- Validation and repair.
- No API key required for tests or CLI help.

### 6. Evidence Agent

Responsibilities:

- Source evidence.
- Evidence verification.
- Hallucination guard.
- Confidence rules.

### 7. Security Agent

Responsibilities:

- Secret redaction.
- No secret leakage to providers.
- No mutation of source files.
- Safe path handling.

### 8. CLI Agent

Responsibilities:

- Typer/Rich CLI.
- `scan`, `build`, `context`, `graph`, `ci`, `health`, `benchmark`, `token-report`.
- JSON output mode.
- Clean exit codes.

### 9. API Agent

Responsibilities:

- FastAPI endpoints.
- Request/response models.
- Project registry resolution.
- Error handling.

### 10. CI/CD Agent

Responsibilities:

- GitHub Actions.
- Package build.
- `pytest`.
- Lint.
- `llmbrain ci`.
- Release workflow.

### 11. Documentation Agent

Responsibilities:

- README.
- `docs/`.
- `examples/`.
- SEO/GEO clarity.
- Open-source onboarding.

### 12. Release Agent

Responsibilities:

- `pyproject.toml`.
- Package metadata.
- Versioning.
- Build and `twine check`.
- Release notes.

## Built-in Coding Agent Personas

The runtime supports running specialized coding agent loops, each with a distinct system prompt, model settings, and security permissions.

### 1. ask (Repository Q&A)
- **Role**: Answers questions about symbols, design, and architecture using the repository memory.
- **Permissions**: Read-only (`SafetyMode.READ_ONLY`).
- **Use Case**: Safe exploration and code intelligence query sessions.

### 2. plan (Analysis and Planning)
- **Role**: Analyzes coding tasks, identifies affected files, lists dependency impact, and outlines test strategies.
- **Permissions**: Read-only (`SafetyMode.READ_ONLY`).
- **Use Case**: Creating architectural blueprints before making modifications.

### 3. build (Implementation and Testing)
- **Role**: Implements code modifications, writes files, applies patches, and executes test suites to solve coding tasks.
- **Permissions**: Ask-before-write or Trusted project (`SafetyMode.ASK_BEFORE_WRITE` / `SafetyMode.TRUSTED_PROJECT`).
- **Use Case**: Active coding work, bug fixing, and package building.

### 4. review (Code and Diff Review)
- **Role**: Reviews current git diff changes, verifying correctness, style, and security against common risks.
- **Permissions**: Read-only (`SafetyMode.READ_ONLY`).
- **Use Case**: Code review stage before commit or pull request integration.

## Agent Collaboration Rules

- Agents must not bypass evidence verification.
- Agents must not replace BrainFrame with JSON prompt blobs.
- Agents must not break mock/offline test mode.
- Agents must not introduce mandatory cloud dependencies.
- Agents must update tests and docs for behavior changes.

## Task Handoff Format

Every agent output should use this format:

- Summary
- Files changed
- Tests run
- Risks
- Follow-up tasks

# SKILLS.md

## Purpose

LLM Brain skills are modular capabilities that transform a codebase into
source-grounded engineering memory.

## Skill Interface Philosophy

Each skill should be:

- As deterministic as possible.
- Source-grounded.
- Testable.
- Provider-independent.
- Mock/offline-compatible.
- CLI/API friendly.

## Core Skills

### scan_project

Input:

- `path`

Output:

- documents
- skipped files
- hashes
- line counts

Purpose: repository and documentation scanning.

### chunk_documents

Input:

- documents

Output:

- chunks with line ranges

Purpose: evidence-preserving chunking.

### redact_secrets

Input:

- document/chunk content

Output:

- redacted content
- redaction count

Purpose: secret masking before LLM context.

### extract_facts

Input:

- chunks or BrainFrame context

Output:

- facts with evidence

Purpose: source-grounded fact extraction.

### extract_entities

Input:

- documents/chunks

Output:

- services, modules, APIs, configs, docs, Docker resources

Purpose: knowledge graph node extraction.

### extract_relations

Input:

- entities + facts as BrainFrame context

Output:

- `depends_on`, `calls`, `imports`, `exposes`, `configured_by`, `related_to`

Purpose: knowledge graph edge extraction.

### verify_evidence

Input:

- facts/relations/wiki pages

Output:

- verified/weak/invalid status

Purpose: hallucination reduction.

### build_brainframe

Input:

- entities, relations, facts

Output:

- compact BrainFrame context

Purpose: token-efficient LLM input.

### generate_wiki

Input:

- verified facts/entities/relations

Output:

- Markdown/MDX pages

Purpose: human-readable engineering memory.

### generate_graph

Input:

- entities/relations

Output:

- `graph.json` / `graph.graphml`

Purpose: knowledge map export.

### build_project

Input:

- path, provider, incremental flag

Output:

- `.llmbrain` artifacts

Purpose: full pipeline orchestration.

### run_ci

Input:

- path, fail_on

Output:

- `ci-result.json`
- exit code

Purpose: CI/CD validation.

### token_report

Input:

- project

Output:

- JSON vs BrainFrame token estimate

Purpose: token efficiency measurement.

## LLM Skills

### structured_fact_extraction

- Uses JSON Schema output.
- Must preserve evidence.

### structured_entity_extraction

- Uses JSON Schema output.
- Must map to stable entity fields.

### structured_relation_extraction

- Uses BrainFrame input context.
- Uses JSON Schema output.
- Must preserve evidence.

### structured_wiki_generation

- Uses JSON Schema output when LLM-backed.
- Must generate evidence-backed pages.

## Skill Safety Rules

- No skill may mutate user source files.
- No skill may send unredacted secrets to LLMs.
- No skill may produce high-confidence claims without evidence.
- No skill may require network access in tests.
- No skill may bypass mock/offline compatibility.

## Adding a New Skill

Checklist:

- Define input/output models.
- Add service implementation.
- Add tests.
- Add CLI/API entry if needed.
- Add docs.
- Ensure mock/offline compatibility.
- Ensure evidence behavior.
- Ensure redaction behavior if LLM context is involved.

## Future Skills

- `docs_drift_detection`
- `pr_comment_generation`
- `github_app_webhook`
- `semantic_search`
- `fts5_search`
- `architecture_diagram_generation`
- `dependency_risk_analysis`
- `incident_memory_compiler`
- `adr_extractor`
- `threat_model_generator`

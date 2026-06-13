# LLMBrain Design Decisions (ADRs)

This document records the architectural and design decisions made while extending LLMBrain into a memory-native coding agent.

## ADR 1: Dual-Purpose `build` Command
- **Context**: The existing command `llmbrain build <path>` executes the static database indexing pipeline and is used in tests and scripts. Phase 7 introduces `llmbrain build "<task>"` to run the coding agent loop.
- **Decision**: Keep the single `build` command but dynamically inspect the first argument. If it is an existing directory path, run the indexer build pipeline. Otherwise, treat it as a coding task and run the Build Agent.
- **Consequence**: Achieves 100% backward compatibility with previous build pipelines and tests, while presenting a clean, unified command interface to the user.

## ADR 2: Schema Extensions for Task Memory
- **Context**: Phase 5 requires storing task summaries, decisions, command history, and failures/resolutions in the database.
- **Decision**: Create four new SQLite tables: `task_runs`, `task_decisions`, `task_commands`, and `task_failures`, with appropriate foreign key mappings to `projects` and `task_runs`.
- **Consequence**: Persistent and queryable record of prior agent decisions, commands, and failures. These are loaded during context assembly to prevent the agent from repeating the same failures.

## ADR 3: Dynamic Tool Schemas via Structured Output Constraint
- **Context**: The agent runtime must call the model and receive structured decisions.
- **Decision**: Implement a Pydantic model `AgentAction` defining a structured schema representing either calling a registered tool with its arguments, or finishing the task with a final response and verification summary. Enforce this via the provider's `generate_structured()` endpoint.
- **Consequence**: Assures type-safe and format-safe model execution loops. Eliminates raw text parsing errors and malformed JSON problems.

## ADR 4: Safety Levels and Permission Prompt Interceptor
- **Context**: Executing shell commands and modifying files requires strict guardrails.
- **Decision**: Implement a `SafetyManager` mapping tool permission levels (`read`, `write`, `shell`, `destructive`) against safety modes:
  - `read-only`
  - `ask-before-write`
  - `trusted-project`
  - `deny-shell`
  Prompts can be intercepted in testing by passing a custom boolean callback (`prompt_func`).
- **Consequence**: Full sandboxing and prompt flexibility. Standardizes interactive command execution safety.

## ADR 5: Rich-based Live Layout TUI Architecture
- **Context**: Phase 5 requires a highly responsive, interactive, and beautifully laid-out terminal user interface (TUI) with multi-pane support, streaming responses, input editing, and dialog modals.
- **Decision**: Implement a custom TUI engine using `rich.live.Live`, `rich.layout.Layout`, and `rich.panel.Panel` combined with an asynchronous keyboard input capture loop reading from non-blocking `sys.stdin`.
- **Consequence**: Provides full layout adaptability, Unicode rendering, and complete control over keyboard focus and modal states. Avoids complex external dependencies like `textual` or `blessings` which might not be packaged in all local development environments, ensuring maximum portability and testability via deterministic programmatic inputs.

## ADR 6: Pure-Python TF-IDF Embedding Fallback
- **Context**: Semantic search requires vector representations of text. External deep learning libraries (like `numpy`, `faiss`, `sentence-transformers`) add hundreds of megabytes of dependencies and make installation/packaging difficult.
- **Decision**: Implement a pure-Python TF-IDF embedder (`TfIdfEmbedder`) with a customizable dimension size, stopwords filtering, and unit-normalized cosine similarity comparison.
- **Consequence**: Zero-dependency, lightweight, fast vector representations and search capabilities that work offline and can easily be upgraded to remote LLM embeddings when available.

## ADR 7: SQLite-backed Asynchronous Job Queue
- **Context**: Running heavy indexing tasks on large repositories can block the agent or main server process.
- **Decision**: Implement a thread-safe, SQLite-backed asynchronous job queue (`IndexQueue`) with priority scheduling (CRITICAL to LOW) and WAL-mode SQLite database.
- **Consequence**: Durable, crash-resistant job execution, atomic dequeue operations, and support for background job tracking.

## ADR 8: Adaptive Concurrency via ResourceManager
- **Context**: Heavy parallel indexing can consume too much CPU and Memory, freezing the developer's system.
- **Decision**: Implement a stdlib-based `ResourceManager` parsing `/proc/stat` and `/proc/meminfo` (falling back to standard values on non-Linux platforms) to adjust concurrent worker count dynamically.
- **Consequence**: Prevents system lockups during large builds by adapting scheduler worker count based on real-time resource utilization.

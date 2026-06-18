# LLM Brain Architecture

## Overview
LLM Brain is a memory-native terminal coding agent and engineering knowledge compiler. It ingests source code, documentation, and metadata, compiling them into a unified knowledge representation (the "Brain") that can be queried and utilized by Large Language Models (LLMs) through a token-efficient BrainFrame context.

## Core Components

1. **Agent Runtime:**
   - Provider-independent multi-agent state machine.
   - Built-in personas (ask, plan, build, review, etc.) with configurable safety boundaries.
   - Comprehensive tool execution framework with auto-retry prevention and output redaction.

2. **Memory Compiler (Services):**
   - Scans repositories with `.gitignore` and binary detection support.
   - Extracts semantic meaning (entities, facts, relations) via LLMs.
   - Generates wiki pages and architecture graphs.

3. **Storage Layer:**
   - **SQLite:** Canonical storage for all structured knowledge (facts, entities, relations, tasks, events).
   - **Vector Store:** Pure-Python SQLite-backed vector store using TF-IDF for semantic search without heavy external dependencies.
   - **JSONL:** Appendable exports of the entire brain for offline processing.

4. **Query Engine & CLI/TUI:**
   - **TUI:** Rich-based interactive terminal UI for managing multi-turn sessions and agent interactions.
   - **BrainFrame:** A highly token-efficient pipe-delimited format (TOON/JTON) that saves up to 80% tokens compared to raw JSON when presenting context to the LLM.

## Data Flow
1. **Source -> Ingestion:** Data is fetched from local Git repositories.
2. **Ingestion -> Compiler:** Code is chunked, facts/entities are extracted via LLMs, and relationships are inferred.
3. **Compiler -> Storage:** TF-IDF embeddings and SQLite records are saved.
4. **Storage -> Query:** When an LLM agent executes a task, the MemoryRetriever fetches the most relevant context and the ContextAssembler formats it into a BrainFrame within the specified token budget.

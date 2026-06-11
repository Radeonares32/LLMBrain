# LLM Brain Architecture

## Overview
LLM Brain is a source-grounded engineering memory compiler. It ingests source code, documentation, incidents, ADRs, and security notes, compiling them into a unified knowledge representation (the "Brain") that can be queried and utilized by Large Language Models (LLMs).

## Core Components

1. **Ingestion Engine:**
   - Scans repositories, documentation sites, and issue trackers.
   - Parses various file formats (Markdown, source code, YAML, etc.).
   - Extracts semantic meaning and metadata.

2. **Memory Compiler:**
   - Processes the raw ingested data.
   - Generates embeddings for semantic search.
   - Constructs an engineering knowledge graph connecting code to architecture, ADRs, and incidents.

3. **Storage Layer:**
   - **Vector Store:** Stores embeddings for fast similarity search.
   - **Graph Database / Relational Store:** Maintains relationships between entities (e.g., a function is connected to an ADR and an incident).

4. **Query Engine & API:**
   - Provides an interface for LLMs or human users to query the memory.
   - Synthesizes context dynamically to ground LLM responses in real, factual data from the repository.

## Data Flow
1. **Source -> Ingestion:** Data is fetched from Git, Jira, Confluence, etc.
2. **Ingestion -> Compiler:** Code is chunked, docs are parsed, and relationships are inferred.
3. **Compiler -> Storage:** Embeddings and graph nodes are saved.
4. **Storage -> Query:** When an LLM needs context, the query engine fetches the most relevant sub-graph and text chunks.

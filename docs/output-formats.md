# Output Formats

When LLM Brain compiles engineering memory or answers a query, it can output data in several formats tailored for different consumers.

## 1. Markdown
The default output format. Human-readable and perfect for rendering in GitHub, Confluence, or internal developer portals.

## 2. JSON / Brainframe JSON
Structured output designed for machine consumption. Ideal when LLM Brain is used as an API backend for another autonomous agent or tool.

```json
{
  "brainframes": [
    {
      "id": "code:src/main.rs",
      "summary": "Main entry point",
      "relations": ["adr:001-rust-adoption"]
    }
  ]
}
```

## 3. Mermaid Diagrams
LLM Brain can automatically synthesize Markdown with embedded Mermaid.js diagrams to visualize architectures, component dependencies, and sequence flows based on the source code and documentation.

## 4. Vector Store Export
The raw embeddings and index can be exported (e.g., as a FAISS index or Chroma DB snapshot) to be loaded into a separate runtime environment.

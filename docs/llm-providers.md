# LLM Providers

LLM Brain is model-agnostic and supports various Large Language Model providers for generating embeddings and synthesizing context.

## Supported Providers

### OpenAI
- **Embeddings:** `text-embedding-3-small`, `text-embedding-3-large`
- **Chat:** `gpt-4o`, `gpt-4-turbo`
- **Config:** Set `OPENAI_API_KEY`.

### Anthropic
- **Chat:** `claude-3-opus`, `claude-3-sonnet`, `claude-3-haiku`
- **Config:** Set `ANTHROPIC_API_KEY`.

### Local / Self-Hosted (Ollama)
For highly sensitive environments, LLM Brain supports local models via Ollama.
- **Embeddings:** `nomic-embed-text`
- **Chat:** `llama3`, `mistral`, `phi3`
- **Config:** Set `OLLAMA_HOST` (defaults to `http://localhost:11434`).

## Adding a New Provider

To add a new provider, implement the `ProviderInterface` in `src/providers/` and register it in the `ProviderFactory`. Please submit an issue using the "LLM Provider Request" template if you'd like a specific provider added to the core project.

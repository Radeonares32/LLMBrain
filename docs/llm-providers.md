# LLM Providers

LLM Brain is model-agnostic and supports various Large Language Model providers for generating embeddings, extracting knowledge, and running coding agents.

## Supported Providers

### DeepSeek (Recommended)
- Highly recommended for cost-effective agent tasks and knowledge extraction.
- **Config:** Set `DEEPSEEK_API_KEY`.

### OpenAI
- **Chat:** `gpt-4o`, `gpt-4-turbo`
- **Config:** Set `OPENAI_API_KEY`.

### Anthropic
- **Chat:** `claude-3-5-sonnet`, `claude-3-opus`, `claude-3-haiku`
- **Config:** Set `ANTHROPIC_API_KEY`.

### Local / Self-Hosted (Ollama)
For highly sensitive environments, LLM Brain supports local models via Ollama.
- **Chat:** `llama3`, `mistral`, `phi3`
- **Config:** Set `OLLAMA_HOST` (defaults to `http://localhost:11434`).

## Adding a New Provider

To add a new provider, extend the `BaseLLMProvider` in `llmbrain/llm/base.py` and register it in the `llmbrain/llm/providers.py` factory. Please submit an issue using the "LLM Provider Request" template if you'd like a specific provider added to the core project.

Note: Deep learning embeddings are currently replaced by a fast, zero-dependency pure-Python TF-IDF embedder (`llmbrain.services.embeddings`) for semantic search, eliminating the need for provider embedding APIs.

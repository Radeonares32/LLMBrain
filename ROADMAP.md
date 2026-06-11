# LLM Brain Roadmap

LLM Brain is a source-grounded engineering memory compiler for repositories, docs, incidents, ADRs, and security notes. This roadmap outlines the high-level goals and direction for the project.

## Q1: Foundation and Core Capabilities
- **Core Engine:** Establish the fundamental memory compiler architecture.
- **Parsers:** Support for Markdown, source code files, and structured logs.
- **Storage Integration:** Local vector store and simple graph representation.
- **Basic CLI:** Initial CLI for compiling memory and querying the "brain".

## Q2: Advanced Integration & Formats
- **Providers Integration:** Support for OpenAI, Anthropic, Gemini, and Local LLMs (Ollama).
- **Format Support:** Complex outputs including Brainframe, Mermaid diagrams, and JSON schemas.
- **CI/CD Pipeline:** Deep integration into GitHub Actions for continuous memory updates.

## Q3: Security, Scale, and Context
- **Security Hardening:** Advanced data sanitization to ensure sensitive IP is scrubbed before LLM processing.
- **Multi-repo Graph:** Cross-repository intelligence and distributed context.
- **Agentic APIs:** Provide an API for external autonomous agents to consume the LLM Brain graph.

*Note: This roadmap is subject to change based on community feedback and project evolution.*

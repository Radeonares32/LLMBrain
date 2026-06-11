# Security Model

LLM Brain processes highly sensitive engineering data, including proprietary source code, vulnerability reports, and infrastructure configurations. Security and data privacy are foundational to the architecture.

## Principles

1. **Data Sanitization First:**
   - Sensitive data (secrets, PII) is stripped before any external LLM processing.
   - Local-first strategies are preferred for highly classified repositories.

2. **Least Privilege Ingestion:**
   - The ingestion engine only requests read access to the specific resources it needs.
   - Write access is strictly disabled for the ingestion pipeline.

3. **Air-gapped Capability:**
   - LLM Brain can run entirely offline using local LLMs (e.g., Ollama) and local vector stores, ensuring zero data egress.

## Threat Model

### Threat: Secret Leakage to Third-party LLMs
**Mitigation:** An aggressive pre-processing pipeline scans for and redacts secrets (e.g., AWS keys, passwords) before text is sent to any external API.

### Threat: Prompt Injection via Ingested Data
**Mitigation:** Ingested content is heavily escaped and separated from system instructions when formatting prompts for the LLM.

### Threat: Unauthorized Access to the Brain
**Mitigation:** The API exposing the LLM Brain requires authentication and enforces RBAC based on the user's permissions in the source system (e.g., GitHub).

# LLMBrain — Test Inventory (TEST_INVENTORY)

This document catalogs the complete test suite of LLMBrain, showing test groups, target modules, execution types, and CI integration.

---

## 1. Test Suite Overview

| Test File | Target Modules | Test Type | Fixture Dependencies | Expected Runtime | CI Inclusion |
| :--- | :--- | :--- | :--- | :--- | :---: |
| `tests/test_agent_runtime.py` | `llmbrain/agent/runtime.py`<br>`llmbrain/agent/safety.py` | Unit / Integration | `temp_project`, `MockRawModelProvider` | ~3.0s | Yes |
| `tests/test_production_tools.py` | `llmbrain/agent/tools.py`<br>`llmbrain/agent/context.py` | Integration / Security | `test_env_dir`, `MockRawModelProvider` | ~0.5s | Yes |
| `tests/test_redactor.py` | `llmbrain/services/redactor.py` | Unit | None | ~0.1s | Yes |
| `tests/test_scanner.py` | `llmbrain/services/scanner.py` | Unit | None | ~0.1s | Yes |
| `tests/test_incremental_build.py` | `llmbrain/services/project_service.py` | Integration | Temp directories | ~1.5s | Yes |
| `tests/test_brainframe.py` | `llmbrain/formats/brainframe.py` | Unit | None | ~0.2s | Yes |
| `tests/test_token_budget.py` | `llmbrain/services/token_budget.py` | Unit | None | ~0.1s | Yes |
| `tests/test_phase6.py` | `llmbrain/core/queue.py`<br>`llmbrain/core/resource_manager.py`<br>`llmbrain/services/profiler.py`<br>`llmbrain/services/remote.py`<br>`llmbrain/services/index_scheduler.py` | Unit / E2E | Temp directories | ~0.5s | Yes |
| `tests/test_phase7.py` | `llmbrain/services/embeddings.py`<br>`llmbrain/storage/vector_store.py`<br>`llmbrain/services/semantic_search.py`<br>`llmbrain/services/multi_repo.py` | Unit / Integration | Temp directories | ~0.5s | Yes |

---

## 2. Test Group Details

### 2.1 Unit Tests
- **State Machine Transitions**: Validates `AgentState` transition constraints (`tests/test_agent_runtime.py`).
- **Secret Redactor Patterns**: Validates regex-based redactions of API keys and private keys (`tests/test_redactor.py` and `tests/test_production_tools.py`).
- **Output Limiter Truncation**: Validates line-based and byte-based head-tail limits (`tests/test_production_tools.py`).
- **BrainFrame Format Builder**: Validates context assembly serialization formatting (`tests/test_brainframe.py`).
- **TfIdfEmbedder & Vector Calculations**: Checks Unit-normalized TF-IDF fallback vector mapping and cosine similarities (`tests/test_phase7.py`).

### 2.2 Integration Tests
- **Workspace Bounds Protection**: Validates path canonicalization, symlink escape rejection, and credential blocks (`tests/test_production_tools.py`).
- **Command Policy Enforcement**: Validates executable allowlist/denylist checks and git arguments restrictions (`tests/test_production_tools.py`).
- **Test Runner Autodetection**: Validates `run_tests` running real temporary tests under Nix or native python environments (`tests/test_agent_runtime.py`).
- **Multi-Repository Registry Operations**: Validates repository entry registration, listing, tagging, and persistence to `registry.json` (`tests/test_phase7.py`).

### 2.3 End-to-End Task Scenarios
- **Scenario A (Repository Q&A)**: End-to-end flow indexing a temporary repository, querying context, and returning an answer (`tests/test_agent_runtime.py`).
- **Scenario C (Approved Code Modification)**: End-to-end tool call requiring and obtaining approval, modifying a file, and succeeding verification (`tests/test_agent_runtime.py`).
- **Scenario D (Denied Write)**: Runs in `SafetyMode.READ_ONLY` mode, rejecting tool calls and auditing the decision (`tests/test_agent_runtime.py`).
- **Scenario F (Malformed Response Recovery)**: Demonstrates the agent loop's bounded JSON parsing error recovery capability (`tests/test_agent_runtime.py`).
- **Scenario G (Repeated Call Protection)**: Verifies the loop terminates and raises `IterationLimitError` when a tool is called repeatedly (`tests/test_agent_runtime.py`).
- **Scenario H (Index Queueing & Concurrency Lifecycle)**: Evaluates background job scheduling, worker pooling, ResourceManager state scaling, and profiler entries logging (`tests/test_phase6.py`).
- **Scenario I (Semantic Retrieval)**: Verifies indexing documents/chunks/facts, creating vector records, and merging results during top-K retrieval query (`tests/test_phase7.py`).

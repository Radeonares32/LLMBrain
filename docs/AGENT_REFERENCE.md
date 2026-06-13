# Agent Reference

This document describes the built-in specialized agents in LLMBrain and their configuration parameters.

## Built-in Agents

### 1. Ask Agent
- **Purpose**: Answer questions about the repository, explain architecture, symbols, and code behavior.
- **Permission Mode**: `read-only`
- **Context Budget**: 16,000 tokens
- **Allowed Tools**: `read_file`, `read_files`, `grep`, `glob`, `git_status`, `git_log`

### 2. Plan Agent
- **Purpose**: Inspect a task, understand affected modules, identify risks, and produce an implementation plan.
- **Permission Mode**: `read-only`
- **Context Budget**: 16,000 tokens
- **Allowed Tools**: `read_file`, `read_files`, `grep`, `glob`, `git_status`, `git_log`

### 3. Build Agent
- **Purpose**: Implement approved coding tasks, modify files, run diagnostics, and run tests.
- **Permission Mode**: `ask-before-write`
- **Context Budget**: 32,000 tokens
- **Allowed Tools**: All read tools, `apply_patch`, `write_file`, `git_status`, `git_diff`, `run_tests`, `diagnostics`
- **Denied Tools**: `delete_file`

### 4. Review Agent
- **Purpose**: Review uncommitted changes or a range of commits.
- **Permission Mode**: `read-only`
- **Context Budget**: 16,000 tokens
- **Allowed Tools**: `read_file`, `grep`, `git_diff`, `git_log`, `git_status`, `diagnostics`

### 5. Debug Agent
- **Purpose**: Reproduce, diagnose, and propose fixes for bugs.
- **Permission Mode**: `ask-before-write`
- **Context Budget**: 24,000 tokens
- **Allowed Tools**: All read tools, `git_diff`, `git_log`, `git_status`, `diagnostics`, `run_tests`

### 6. Test Agent
- **Purpose**: Inspect coverage gaps, write tests, and run tests.
- **Permission Mode**: `ask-before-write`
- **Context Budget**: 24,000 tokens
- **Allowed Tools**: All read tools, `git_diff`, `git_log`, `git_status`, `diagnostics`, `run_tests`, `write_file`

### 7. Security Agent
- **Purpose**: Perform secure code reviews and audit authentication/authorization.
- **Permission Mode**: `read-only`
- **Context Budget**: 24,000 tokens
- **Allowed Tools**: All read tools, `git_diff`, `git_log`, `git_status`, `diagnostics`, `run_tests`

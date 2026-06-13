# Security Policy

## Supported Versions

LLM Brain is actively maintained. Security updates are provided for the latest major version.

| Version | Supported          |
| ------- | ------------------ |
| 1.x.x   | :white_check_mark: |
| < 1.0   | :x:                |

## Reporting a Vulnerability

Security is a top priority for LLM Brain, as we process sensitive repositories, docs, incidents, ADRs, and security notes.

If you discover a security vulnerability within LLM Brain, please do not disclose it publicly. Instead, send an email to the core maintainers. We will respond to your report as soon as possible and work with you to resolve the issue.

Please provide the following information in your report:
- A description of the vulnerability.
- Steps to reproduce the vulnerability.
- Any potential impact.
- Any suggested mitigations.

We will keep you informed of our progress and will publicly disclose the vulnerability once a fix has been released.

## Agent Security & Permission Policies

With the extension of LLMBrain to a coding agent, the following security constraints are enforced at the runtime level:

### 1. Safety Permission Modes
Users can select from four runtime safety modes to control tool execution risk:
- **`read-only`**: Only read-only tools are allowed. File writes, shell commands, and git modifications are completely blocked.
- **`ask-before-write`**: (Default) Allows read operations. Prompts the user in the terminal for approval before executing file writes, patches, shell commands, or destructive git tasks.
- **`trusted-project`**: Automatically permits file reads/writes. Prompts the user before executing shell commands or destructive operations.
- **`deny-shell`**: Blocks shell commands entirely. Prompts the user for write operations.

### 2. Path Traversal Prevention
All file tools (`read_file`, `write_file`, `apply_patch`) validate target paths. Access to files outside the workspace root is rejected with a `PermissionError` to prevent unauthorized file access.

### 3. Destructive Operation Blocklist
Commands containing dangerous operations (e.g. `rm -rf /`, force push options `push -f`, resetting commits `reset --hard`) are flagged and blocklisted, requiring explicit confirmation or blocked depending on the current safety settings.


# LLMBrain — Security Validation Report (SECURITY_VALIDATION)

This report documents the security audit, threat model, tested safety controls, and identified boundary conditions for the LLMBrain agent runtime and production tool execution layer.

---

## 1. Threat Model

LLMBrain executes tools inside the user's repository workspace. Because it is powered by an LLM that can generate arbitrary code or commands, the main threats are:
1. **Workspace Escapes**: The model attempts to read, write, or delete files outside the workspace root (e.g. `/etc/passwd` or user home credentials).
2. **Credential Theft**: The model reads sensitive local development secrets (e.g. `.env`, `.ssh/id_rsa`, or `config.json` files).
3. **Arbitrary Command Injection**: The model attempts shell execution with chained commands, piping, redirecting, or invoking dangerous executables (`curl`, `rm -rf`, etc.).
4. **Secret Leakage**: Local secrets (API keys, connection strings, private keys) are printed to stdout/stderr and leaked back to the LLM provider API.

---

## 2. Security Controls & Protections

### 2.1 Workspace Sandboxing and Path Safety
- **PathResolver**: All relative and absolute paths passed to file operations (`read_file`, `write_file`, `glob`, `grep`, `delete_file`) are canonicalized using `Path.resolve()`. 
- **Escape Rejection**: Any path resolving outside `workspace_root` raises a `ValueError` ("Path escapes the workspace boundary").
- **Symlink Protection**: Symbolic links pointing outside the workspace boundary are actively detected and rejected.
- **Credential Path Restrictions**: Direct or indirect access to `.env` files, `.ssh/` directory, private keys (e.g., `id_rsa`, `id_ed25519`), or `credentials`/`config.json` files is blocked with a `PermissionError`.

### 2.2 Shell Command Execution Policy
- **Structured Spawning**: Commands are spawned as argument lists (`subprocess.Popen` with list of strings). This completely mitigates traditional string-based shell injections.
- **Operator Blocking**: Dangerous shell operators (like `;`, `&&`, `|`, `>`, `<`, `$()`) are not processed or are rejected by splitting.
- **CommandPolicy**: Executables are categorized. Destructive binaries (like `rm`, `sudo`, `su`, `chmod`, `docker`) or network tools (`curl`, `wget`) are rejected outright.
- **Git Restrictions**: Prohibited git subcommands (like `git push`, `git reset`, `git clean`, `git rebase`, `git force`) are blocked at the policy level.

### 2.3 Secret Protection (Redactor)
- **SecretRedactor**: Automatically monitors stdout, stderr, and arguments of all executed tools.
- Regular expression patterns redact API keys, passwords, database connection strings, bearer tokens, and private key blocks (replacing them with `[REDACTED]`).

### 2.4 Output Truncation
- **OutputLimiter**: Protects the agent loop from context-budget exhaustion or denial of service caused by reading giant files or spawning long-running verbose processes. Outputs are truncated cleanly using head-tail compression.

---

## 3. Remaining Limitations & Risks

1. **No OS-level Container Sandboxing**: LLMBrain runs directly in the host user's environment. While path and command filters are active, it does not use Docker or gVisor to isolate execution at the OS kernel level.
2. **Platform-Specific Paths**: Path resolving on Windows vs. Posix hosts differs in path separators (`\` vs `/`). Mixed separators are checked, but native sandboxing relies on Python's path canonicalization.
3. **Indirect Write Operations**: Execution of compilers or build scripts (like `pytest` or `cargo build`) could potentially trigger secondary code execution if a test file itself contains malicious code. Therefore, trusted project mode must only be run on trusted repositories.

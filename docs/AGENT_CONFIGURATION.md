# Agent Configuration

You can configure built-in agents and define custom agents in LLMBrain.

## Configuration Precedence

Precedence is resolved in this order:
1. Built-in defaults
2. User configuration
3. Project configuration (`.llmbrain/agents.yaml`)
4. Explicit CLI overrides

## Project-Level Configuration (`.llmbrain/agents.yaml`)

Example configuration defining a custom database migration planning agent extending `plan`:

```yaml
agents:
  migration-planner:
    description: Plans database migrations without applying them
    extends: plan
    context:
      token_budget: 20000
    tools:
      allow:
        - read_file
        - grep
        - git_log
```

## Schema Validation

All custom agent definitions must conform to:
- Name uniqueness.
- Valid prompt references.
- Bounded iterations and tool call limits.
- No escalation of default global system safety parameters.

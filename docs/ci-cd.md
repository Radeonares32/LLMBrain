# CI/CD Integration

LLM Brain is designed to be fully integrated into your Continuous Integration and Continuous Deployment (CI/CD) pipelines to ensure the engineering memory is always up to date.

## GitHub Actions

You can use the official LLM Brain GitHub Action to compile memory on every push to the main branch.

```yaml
name: Compile LLM Brain Memory

on:
  push:
    branches:
      - main

jobs:
  update-brain:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout code
        uses: actions/checkout@v3

      - name: Run LLM Brain Compiler
        uses: llmbrain/compiler-action@v1
        with:
          repository_path: .
          llm_provider: openai
          api_key: ${{ secrets.OPENAI_API_KEY }}
```

## Continuous Updates

1. **Incremental Compilation:** The CI/CD step uses Git diffs to only recompile modified files, saving time and API costs.
2. **Pull Request Context:** LLM Brain can be run in "Dry Run" mode on PRs to provide the reviewer with synthesized context about how the PR affects the broader system.

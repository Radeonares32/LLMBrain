# Releasing LLM Brain

To cut a new release:

1. **Version Bump**: Update `__version__` in `llmbrain/__init__.py` and `version` in `pyproject.toml`.
2. **Run Tests**:
   ```bash
   python -m pip install -e ".[dev]"
   pytest
   ```
3. **Build Package**:
   ```bash
   python -m build
   ```
4. **Check Package**:
   ```bash
   twine check dist/*
   ```
5. **Create Git Tag**:
   ```bash
   git tag v0.1.0
   git push origin v0.1.0
   ```
6. **Publish to PyPI**:
   The `.github/workflows/release.yml` action will automatically build and publish to PyPI on new tags.

7. **GitHub Release Notes**:
   GitHub Action will auto-generate release notes based on merged PRs.

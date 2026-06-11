import os

import pytest

from llmbrain.services.project_service import ProjectService


def test_sample_project_build():
    provider = os.getenv("LLMBRAIN_TEST_PROVIDER", "openai")
    api_key_names = {
        "openai": "OPENAI_API_KEY",
        "deepseek": "DEEPSEEK_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
    }
    key_name = api_key_names.get(provider)
    if key_name and not os.getenv(key_name):
        pytest.skip(f"{key_name} is required for production build test")
    if provider == "ollama" and not os.getenv("OLLAMA_MODEL"):
        pytest.skip("OLLAMA_MODEL is required for production build test")

    service = ProjectService()
    result = service.build_project("examples/sample-project", llm_provider=provider)
    assert result is not None

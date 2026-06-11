from pathlib import Path


def test_agent_guidance_files_exist_and_are_not_empty():
    root = Path(__file__).resolve().parents[1]

    for filename in ["CLAUDE.md", "AGENTS.md", "SKILLS.md"]:
        path = root / filename
        assert path.exists()
        assert path.read_text(encoding="utf-8").strip()

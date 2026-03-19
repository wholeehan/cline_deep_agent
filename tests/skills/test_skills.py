"""Tests for SKILL.md files — validation and loading."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

SKILLS_DIR = Path(__file__).parent.parent.parent / "skills"

EXPECTED_SKILLS = ["task_decomposition", "qa_verification", "decision_policy", "cline_qa"]


class TestSkillFiles:
    """Validate that all SKILL.md files exist and have correct frontmatter."""

    @pytest.mark.parametrize("skill_name", EXPECTED_SKILLS)
    def test_skill_md_exists(self, skill_name: str) -> None:
        skill_path = SKILLS_DIR / skill_name / "SKILL.md"
        assert skill_path.exists(), f"Missing SKILL.md for {skill_name}"

    @pytest.mark.parametrize("skill_name", EXPECTED_SKILLS)
    def test_skill_has_valid_frontmatter(self, skill_name: str) -> None:
        skill_path = SKILLS_DIR / skill_name / "SKILL.md"
        content = skill_path.read_text()

        # Extract YAML frontmatter
        assert content.startswith("---"), f"{skill_name} SKILL.md must start with ---"
        end = content.index("---", 3)
        frontmatter = yaml.safe_load(content[3:end])

        assert "name" in frontmatter, f"{skill_name} missing 'name' in frontmatter"
        assert "description" in frontmatter, f"{skill_name} missing 'description' in frontmatter"
        assert len(frontmatter["name"]) <= 64, f"{skill_name} name too long"
        assert len(frontmatter["description"]) <= 1024, f"{skill_name} description too long"

    @pytest.mark.parametrize("skill_name", EXPECTED_SKILLS)
    def test_skill_has_when_to_use_section(self, skill_name: str) -> None:
        skill_path = SKILLS_DIR / skill_name / "SKILL.md"
        content = skill_path.read_text()
        assert "## When to Use" in content, f"{skill_name} missing '## When to Use' section"

    @pytest.mark.parametrize("skill_name", EXPECTED_SKILLS)
    def test_skill_has_instructions_section(self, skill_name: str) -> None:
        skill_path = SKILLS_DIR / skill_name / "SKILL.md"
        content = skill_path.read_text()
        assert "## Instructions" in content or "## Policy Table" in content, \
            f"{skill_name} missing instructions or policy section"


class TestDecisionPolicy:
    """Specific tests for the decision policy skill."""

    def test_policy_classifies_file_writes_as_auto_approve(self) -> None:
        content = (SKILLS_DIR / "decision_policy" / "SKILL.md").read_text()
        assert "Auto-approve" in content
        assert "write_file" in content or "File write" in content

    def test_policy_classifies_http_as_escalate(self) -> None:
        content = (SKILLS_DIR / "decision_policy" / "SKILL.md").read_text()
        assert "Escalate" in content
        assert "HTTP" in content or "curl" in content

    def test_policy_classifies_delete_as_always_escalate(self) -> None:
        content = (SKILLS_DIR / "decision_policy" / "SKILL.md").read_text()
        assert "Always escalate" in content
        assert "delete" in content.lower()


class TestTaskDecomposition:
    """Specific tests for the task decomposition skill."""

    def test_output_format_includes_required_fields(self) -> None:
        content = (SKILLS_DIR / "task_decomposition" / "SKILL.md").read_text()
        for field in ["title", "context", "acceptance_criterion", "complexity", "depends_on"]:
            assert field in content, f"Missing field '{field}' in task_decomposition SKILL.md"


class TestQaVerification:
    """Specific tests for the QA verification skill."""

    def test_has_verified_and_failed_verdicts(self) -> None:
        content = (SKILLS_DIR / "qa_verification" / "SKILL.md").read_text()
        assert "verified" in content
        assert "failed" in content


class TestClineQa:
    """Specific tests for the cline QA skill."""

    def test_instructs_grounded_answers(self) -> None:
        content = (SKILLS_DIR / "cline_qa" / "SKILL.md").read_text()
        assert "context" in content.lower()
        assert "escalate" in content.lower()

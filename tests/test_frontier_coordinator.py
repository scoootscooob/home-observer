"""The sandboxed frontier planner is standard-library only and parses plans defensively."""
import ast
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import frontier_coordinator  # noqa: E402

STDLIB = set(sys.stdlib_module_names)


def test_coordinator_imports_only_the_standard_library():
    tree = ast.parse((ROOT / "scripts/frontier_coordinator.py").read_text())
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert imported <= STDLIB, imported - STDLIB
    assert "home_observer" not in imported


def test_plan_parsing_accepts_fenced_json_and_rejects_non_objects():
    plan = frontier_coordinator.parse_plan('```json\n{"commands": [], "assessment": {"recipe_step_complete": false}}\n```')
    assert plan["commands"] == [] and plan["assessment"]["recipe_step_complete"] is False
    prefixed = frontier_coordinator.parse_plan('Sure. {"commands": [{"type": "ack_export"}], "wait": true} done')
    assert prefixed["commands"][0]["type"] == "ack_export"
    with pytest.raises(ValueError):
        frontier_coordinator.parse_plan("no json here")
    with pytest.raises(ValueError):
        frontier_coordinator.parse_plan('{"commands": "not a list"}')
    with pytest.raises(ValueError):
        frontier_coordinator.parse_plan('{"commands": [1,]}')


def test_system_prompt_forbids_media_and_completes_only_on_verification():
    prompt = frontier_coordinator.SYSTEM_PROMPT
    assert "never see images" in prompt and "must never ask for any" in prompt
    assert "ONLY when a verification_result export says" in prompt and 'verdict "confirmed"' in prompt
    assert "ack_export" in prompt and "verify" in prompt and "create_watch" in prompt

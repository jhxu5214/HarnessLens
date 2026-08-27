import json
import os
import tempfile
from pathlib import Path

import pytest

from harnesslens.evolution.analyzer import AnalyzerModule
from harnesslens.core.budget import CreationBudget
from harnesslens.core.config import load_repo_env
from harnesslens.evolution.discovery import DiscoveryModule


REPO_ROOT = Path(__file__).resolve().parents[1]


pytestmark = pytest.mark.skipif(
    os.environ.get("HAI_RUN_HARNESS_QUERY_LIVE") != "1",
    reason="set HAI_RUN_HARNESS_QUERY_LIVE=1 to run Query-to-Analyzer model probes",
)


@pytest.mark.parametrize("harness", ["opencode", "pi", "codex"])
def test_same_harness_query_to_analyzer_candidate_and_cleanup(harness):
    load_repo_env(REPO_ROOT)
    assert os.environ.get("DEEPSEEK_API_KEY")
    temporary_root = None
    with tempfile.TemporaryDirectory(prefix=f"harnesslens-{harness}-query-e2e-") as raw:
        temporary_root = Path(raw)
        run_root = temporary_root / "run"
        budget = CreationBudget(
            run_root / "creation_budget.json", total=10, baseline_used=0
        )
        discovery = DiscoveryModule(
            repo_root=REPO_ROOT,
            run_root=run_root,
            budget=budget,
            harness=harness,
        )
        public_environment = {
            "tool_transport": {"kind": "mcp", "server_id": "query-server"},
            "tools": [
                {
                    "name": "lookup_record",
                    "description": "Look up one record by exact identifier.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "record_id": {
                                "type": "string",
                                "description": "Exact record identifier.",
                            }
                        },
                        "required": ["record_id"],
                    },
                }
            ],
        }
        query_input = discovery._harness_query_input(
            {"environment": public_environment}
        )
        query_result = discovery._run_harness_query(query_input)

        assert query_result.output is not None, query_result.validation_error
        assert query_result.harness == harness
        assert query_result.output["harness"] == harness
        assert query_result.output["modifiable_modules"], query_result.output
        if harness == "opencode":
            modules = {
                item["id"]: item
                for item in query_result.output["modifiable_modules"]
            }
            assert "instructions_rules" in modules, json.dumps(
                query_result.output,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            assert modules["instructions_rules"]["edit_contract"] == {
                "scope": "project",
                "path": "opencode.json",
                "mechanism": "config",
                "key": "instructions",
            }, query_result.output
        elif harness in {"pi", "codex"}:
            modules = {
                item["id"]: item
                for item in query_result.output["modifiable_modules"]
            }
            assert "project_instructions" in modules, json.dumps(
                query_result.output,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            assert modules["project_instructions"]["edit_contract"] == {
                "scope": "project",
                "path": "AGENTS.md",
                "mechanism": "file",
            }, query_result.output

        analyzer = AnalyzerModule(
            run_root=run_root,
            budget=budget,
            harness=harness,
        )
        output = analyzer._run_side(
            label="query-probe",
            side="reusable",
            experiences=[
                {
                    "id": "exp-query-probe",
                    "summary": "Preserve exact-identifier lookup behavior.",
                    "evidence_refs": ["trajectory:query-probe"],
                    "trigger": "A task supplies one exact record identifier.",
                    "procedure": [
                        "Pass the supplied identifier unchanged to lookup_record.",
                        "Check the returned identifier before using the record.",
                    ],
                }
            ],
            discovery={
                "harness_query": query_result.output,
                "task_categories": {
                    "categories": [
                        {
                            "id": "exact-lookup",
                            "name": "Exact lookup",
                            "purpose": "Retrieve a record by an exact identifier.",
                            "task_ids": ["query-probe"],
                        }
                    ],
                    "notes": [],
                },
                "public_environment": public_environment,
            },
        )

        assert output["candidates"]
        assert output["coverage"]["experience_ids"] == ["exp-query-probe"]
        manifests = list((run_root / "intelligent_jobs").glob("*/interaction_manifest.json"))
        assert len(manifests) >= 2
        assert {
            json.loads(path.read_text(encoding="utf-8"))["harness"]
            for path in manifests
        } == {harness}

    assert temporary_root is not None
    assert not temporary_root.exists()

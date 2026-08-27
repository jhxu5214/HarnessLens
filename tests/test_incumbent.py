import json

import pytest

from harnesslens.evolution.incumbent import load_incumbent_candidate


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_loads_train_accepted_opencode_incumbent(tmp_path):
    snapshot = (
        tmp_path
        / "rollout_evidence"
        / "old-run"
        / "versions_percell"
        / "bird_mini_dev_challenging"
        / "candidate-04"
    )
    _write_json(
        snapshot / "harness" / "opencode" / "patch.json",
        {
            "config_patch": {},
            "files": [],
            "instructions": [],
            "prompt_appends": [],
        },
    )
    _write_json(
        snapshot / "harness" / "opencode" / "patch_descs.json",
        {
            "execute_sql": {
                "desc": "Use CAST(... AS REAL) for integer division."
            }
        },
    )
    submission = tmp_path / "submission" / "final.json"
    _write_json(
        submission,
        {
            "selected_version": "candidate-04",
            "snapshot_path": str(snapshot),
            "iteration_history": [
                {
                    "candidate_id": "revision-candidate-03",
                    "candidate_version": "candidate-04",
                    "review_decision": "accept_delta",
                    "selected_version": "candidate-04",
                    "rollout_task_ids": ["bird_760", "bird_775", "outside"],
                    "channel_diffs": [
                        {
                            "channel_id": "tool_description",
                            "experience_ids": ["percentage-numeric"],
                        }
                    ],
                    "review_evidence": {
                        "recovered_task_ids": ["bird_760"],
                        "attributable_regression_task_ids": [],
                    },
                }
            ],
        },
    )

    candidate = load_incumbent_candidate(
        submission,
        cell="bird_mini_dev_challenging",
        harness="opencode",
        train_task_ids=("bird_760", "bird_775"),
    )

    assert candidate["_portfolio_side"] == "incumbent"
    assert candidate["_direct_task_ids"] == ["bird_760", "bird_775"]
    assert candidate["channel_plan"] == [
        {
            "channel_id": "tool_description",
            "operation": "revalidate a previously TRAIN-accepted cumulative artifact",
            "experience_ids": ["percentage-numeric"],
            "rationale": "Retest prior attributable TRAIN evidence under the current run.",
        }
    ]
    assert candidate["manifest_delta"]["tool_desc_patches"] == {
        "execute_sql": {"desc": "Use CAST(... AS REAL) for integer division."}
    }
    assert candidate["_prior_train_evidence"]["recovered_task_ids"] == ["bird_760"]


def test_loads_native_incumbent_manifest(tmp_path):
    snapshot = (
        tmp_path
        / "versions_percell"
        / "retail"
        / "candidate-02"
    )
    _write_json(
        snapshot / "harness" / "pi" / "manifest.json",
        {
            "config_patch": {},
            "files": [],
            "instructions": [],
            "prompt_appends": ["Confirm irreversible actions before execution."],
            "tool_desc_patches": {},
        },
    )
    submission = tmp_path / "final.json"
    _write_json(
        submission,
        {
            "selected_version": "candidate-02",
            "snapshot_path": str(snapshot),
            "iteration_history": [
                {
                    "candidate_id": "confirmation",
                    "candidate_version": "candidate-02",
                    "review_decision": "accept_delta",
                    "selected_version": "candidate-02",
                    "rollout_task_ids": ["retail_1"],
                    "channel_diffs": [
                        {"channel_id": "system_prompt", "experience_ids": ["exp-1"]}
                    ],
                    "review_evidence": {"recovered_task_ids": ["retail_1"]},
                }
            ],
        },
    )

    candidate = load_incumbent_candidate(
        submission,
        cell="retail",
        harness="pi",
        train_task_ids=("retail_1",),
    )

    assert candidate["manifest_delta"]["prompt_appends"] == [
        "Confirm irreversible actions before execution."
    ]


def test_rejects_incumbent_from_another_cell(tmp_path):
    snapshot = tmp_path / "versions_percell" / "retail" / "candidate-01"
    _write_json(
        snapshot / "harness" / "opencode" / "patch.json",
        {"instructions": ["A prior rule."]},
    )
    submission = tmp_path / "final.json"
    _write_json(
        submission,
        {
            "selected_version": "candidate-01",
            "snapshot_path": str(snapshot),
            "iteration_history": [],
        },
    )

    with pytest.raises(ValueError, match="cell"):
        load_incumbent_candidate(
            submission,
            cell="banking_knowledge",
            harness="opencode",
            train_task_ids=("banking_1",),
        )

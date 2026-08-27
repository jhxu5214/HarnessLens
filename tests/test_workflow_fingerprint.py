import pytest

from harnesslens.core.workflow_fingerprint import (
    assert_workflow_fingerprint,
    establish_workflow_fingerprint,
)


def _workflow_tree(tmp_path):
    repo = tmp_path / "repo"
    package = repo / "harnesslens" / "evolution"
    package.mkdir(parents=True)
    (repo / "run_e2e.py").write_text("main = 1\n")
    (package / "module.py").write_text("VALUE = 1\n")
    return repo


def test_workflow_fingerprint_rejects_source_change_during_resume(tmp_path):
    repo = _workflow_tree(tmp_path)
    run = repo / "runs" / "train" / "run-a"
    run.mkdir(parents=True)
    fingerprint = establish_workflow_fingerprint(repo_root=repo, run_root=run)

    assert_workflow_fingerprint(
        repo_root=repo,
        run_root=run,
        expected_sha256=fingerprint["sha256"],
    )
    (repo / "harnesslens" / "evolution" / "module.py").write_text(
        "VALUE = 2\n"
    )

    with pytest.raises(RuntimeError, match="source changed during the run"):
        assert_workflow_fingerprint(
            repo_root=repo,
            run_root=run,
            expected_sha256=fingerprint["sha256"],
        )


def test_legacy_progress_cannot_be_adopted_without_a_fingerprint(tmp_path):
    repo = _workflow_tree(tmp_path)
    run = repo / "runs" / "train" / "run-a"
    (run / "main_agent").mkdir(parents=True)

    with pytest.raises(RuntimeError, match="cannot be safely resumed"):
        establish_workflow_fingerprint(repo_root=repo, run_root=run)

from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW_PATH = REPOSITORY_ROOT / ".github/workflows/plugin_submission_orchestrator.yml"
HELPER_PATH = REPOSITORY_ROOT / "brainscore_language/submission/actions_helpers.py"


def test_metadata_update_jenkins_trigger_is_not_exposed():
    helper = HELPER_PATH.read_text()

    assert "def trigger_update_existing_metadata" not in helper
    assert "'trigger_update_existing_metadata'" not in helper


def test_metadata_only_merges_do_not_request_jenkins_updates():
    workflow = WORKFLOW_PATH.read_text()
    post_merge_condition = workflow.split("  post_merge_scoring:\n", 1)[1].split(
        "    runs-on:", 1
    )[0]

    assert "trigger_update_existing_metadata" not in workflow
    assert "/job/update_existing_metadata" not in workflow
    assert "needs.detect_changes.outputs.needs_scoring == 'true'" in post_merge_condition
    assert "needs.detect_changes.outputs.metadata_only == 'false'" in post_merge_condition
    assert "needs.detect_changes.outputs.metadata_only == 'true'" not in post_merge_condition


def test_post_merge_job_has_no_metadata_only_steps():
    workflow = WORKFLOW_PATH.read_text()
    post_merge_job = workflow.split("  post_merge_scoring:\n", 1)[1].split(
        "  notify_on_failure:\n", 1
    )[0]

    assert "metadata_only == 'true'" not in post_merge_job

"""Exercise workflow code selection with real, isolated upstream/fork Git refs.

WORKFLOW_UNDER_TEST can select an older YAML file for a red/green regression
check. No GitHub, Jenkins, AWS, or network requests are made by these tests.
"""

import importlib.util
import os
from pathlib import Path
import re
import subprocess
from types import SimpleNamespace

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".github/workflows/plugin_submission_orchestrator.yml"
HELPER = Path("brainscore_language/submission/actions_helpers.py")
CONTEXTS = [
    "Language Unittests, Plugins", "Language Unittests, Non-Plugins",
    "Language Integration Tests", "docs/readthedocs.org:brain-score-language",
]


def git(directory, *args):
    return subprocess.check_output(
        ["git", "-C", str(directory), *args], text=True, stderr=subprocess.PIPE,
    ).strip()


def commit(directory, subject):
    git(directory, "add", ".")
    git(directory, "-c", "user.name=CI Fixture", "-c", "user.email=fixture@example.test",
        "commit", "-m", subject)
    return git(directory, "rev-parse", "HEAD")


@pytest.fixture
def repositories(tmp_path):
    upstream = tmp_path / "upstream"
    upstream.mkdir()
    git(upstream, "init", "-b", "main")
    helper = upstream / HELPER
    helper.parent.mkdir(parents=True)
    # A deliberately obsolete fixture makes accidental selection observable.
    # Correct selection must execute the real current helper below instead.
    helper.write_text("def validate_pr(*args, **kwargs):\n    return {'all_tests_pass': True}\n")
    original_base = commit(upstream, "test(ci): add obsolete validator fixture")
    helper.write_text((ROOT / HELPER).read_text())
    current_base = commit(upstream, "fix(ci): install current validator fixture")

    fork = tmp_path / "fork"
    git(tmp_path, "clone", "--no-hardlinks", str(upstream), str(fork))
    (fork / HELPER).write_text("raise AssertionError('Untrusted fork helper executed')\n")
    fork_head = commit(fork, "test(ci): replace helper in untrusted fork fixture")
    return SimpleNamespace(upstream=upstream, fork=fork, original_base=original_base,
                           current_base=current_base, fork_head=fork_head, directory=tmp_path)


def resolve(value, context):
    """Expand a full dotted-path expression used by actions/checkout inputs."""
    if not isinstance(value, str):
        return value
    match = re.fullmatch(r"\$\{\{\s*([\w.]+)\s*\}\}", value)
    if not match:
        return value
    for part in match.group(1).split("."):
        context = context[part]
    return context


@pytest.fixture
def checked_out(repositories):
    repos = repositories
    context = {"github": {
        "repository": "upstream/language",
        "event": {"pull_request": {
            "base": {"sha": repos.original_base, "ref": "main"},
            "head": {"sha": repos.fork_head, "ref": "main",
                     "repo": {"full_name": "contributor/language"}},
        }},
    }}
    workflow = yaml.safe_load(Path(os.environ.get("WORKFLOW_UNDER_TEST", WORKFLOW)).read_text())
    inputs = workflow["jobs"]["validate_pr"]["steps"][0]["with"]
    repository = resolve(inputs.get("repository", "${{ github.repository }}"), context)
    source = {"upstream/language": repos.upstream, "contributor/language": repos.fork}[repository]
    ref = resolve(inputs["ref"], context)
    checkout = repos.directory / "checkout"
    git(repos.directory, "clone", "--no-checkout", str(source), str(checkout))
    git(checkout, "checkout", "--detach", ref)
    spec = importlib.util.spec_from_file_location("checked_out_validator", checkout / HELPER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return repos, checkout, module


def validate(checked_out, monkeypatch, *, missing=False, changed_head=False):
    repos, _, helper = checked_out

    def api(url, token):
        if "/commits/" in url:
            assert repos.fork_head in url
            return {"statuses": [] if missing else [
                {"context": context, "state": "success", "updated_at": "2026-11-08T00:00:00Z"}
                for context in CONTEXTS
            ]}
        assert url.endswith("/pulls/413")
        return {"head": {"sha": "b" * 40 if changed_head else repos.fork_head}, "labels": []}

    monkeypatch.setattr(helper, "get_data", api, raising=False)
    return helper.validate_pr(413, repos.fork_head, False, None, max_wait_time=0)


def test_old_pr_selects_current_upstream_code(checked_out):
    repos, checkout, _ = checked_out
    assert repos.original_base != repos.current_base != repos.fork_head
    assert git(checkout, "rev-parse", "HEAD") == repos.current_base


def test_selected_code_rejects_missing_statuses(checked_out, monkeypatch):
    assert validate(checked_out, monkeypatch, missing=True)["all_tests_pass"] is False


def test_selected_code_checks_fork_head_without_executing_fork_helper(checked_out, monkeypatch):
    assert validate(checked_out, monkeypatch)["all_tests_pass"] is True


def test_selected_code_rejects_new_push_during_validation(checked_out, monkeypatch):
    assert validate(checked_out, monkeypatch, changed_head=True)["all_tests_pass"] is False

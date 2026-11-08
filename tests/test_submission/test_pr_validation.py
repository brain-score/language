from unittest.mock import Mock

import pytest

from brainscore_language.submission import actions_helpers


CONTEXTS = [
    "Language Unittests, Plugins",
    "Language Unittests, Non-Plugins",
    "Language Integration Tests",
    "docs/readthedocs.org:brain-score-language",
]
HEAD = "a" * 40


def statuses(**changes):
    return [
        {"context": context, "state": changes.get(context, "success"),
         "updated_at": "2026-11-08T10:00:00Z"}
        for context in CONTEXTS if changes.get(context, "success") is not None
    ]


def prepare_api(monkeypatch, polls, current_head=HEAD):
    remaining = iter(polls)

    def get_data(url, token):
        if f"/commits/{HEAD}/status?per_page=100" in url:
            return {"statuses": next(remaining)}
        assert url.endswith("/pulls/413")
        return {"head": {"sha": current_head}, "labels": [{"name": "submission_prepared"}]}

    api = Mock(side_effect=get_data)
    monkeypatch.setattr(actions_helpers, "get_data", api)
    return api


def validate(**kwargs):
    return actions_helpers.validate_pr(413, HEAD, False, "read-token", max_wait_time=0, **kwargs)


def test_all_required_successes_pass(monkeypatch):
    prepare_api(monkeypatch, [statuses()])
    result = validate()
    assert result["all_tests_pass"] is True
    assert result["head_matches"] is True


@pytest.mark.parametrize("context", CONTEXTS)
@pytest.mark.parametrize("state", [None, "pending", "failure", "error"])
def test_each_required_status_must_succeed_even_after_timeout(monkeypatch, context, state):
    prepare_api(monkeypatch, [statuses(**{context: state})])
    result = validate()
    assert result["all_tests_pass"] is False
    assert result["is_automergeable"] is False
    assert result["test_results"][context] == state


def test_no_reported_tests_never_pass(monkeypatch):
    prepare_api(monkeypatch, [[]])
    result = validate()
    assert result["all_tests_pass"] is False
    assert result["is_automergeable"] is False


def test_new_push_invalidates_previous_head_results(monkeypatch):
    prepare_api(monkeypatch, [statuses()], current_head="b" * 40)
    result = validate()
    assert result["head_matches"] is False
    assert result["all_tests_pass"] is False
    assert result["is_automergeable"] is False


def test_pending_status_can_finish_before_timeout(monkeypatch):
    prepare_api(monkeypatch, [statuses(**{CONTEXTS[0]: "pending"}), statuses()])
    monkeypatch.setattr(actions_helpers.time, "monotonic", lambda: 0)
    sleep = Mock()
    monkeypatch.setattr(actions_helpers.time, "sleep", sleep)
    result = actions_helpers.validate_pr(413, HEAD, False, "read-token", max_wait_time=60)
    assert result["all_tests_pass"] is True
    sleep.assert_called_once_with(30)


def test_known_failure_returns_without_wait_for_missing_statuses(monkeypatch):
    prepare_api(monkeypatch, [statuses(**{CONTEXTS[0]: "failure", CONTEXTS[1]: None})])
    sleep = Mock(side_effect=AssertionError("A failed run must not wait"))
    monkeypatch.setattr(actions_helpers.time, "sleep", sleep)
    result = actions_helpers.validate_pr(413, HEAD, False, "read-token", max_wait_time=60)
    assert result["all_tests_pass"] is False
    sleep.assert_not_called()


def test_latest_status_wins_over_old_success():
    history = [
        {"context": CONTEXTS[0], "state": "success", "updated_at": "2026-11-08T10:00:00Z"},
        {"context": CONTEXTS[0], "state": "failure", "updated_at": "2026-11-08T11:00:00Z"},
    ]
    assert actions_helpers.get_statuses_result(CONTEXTS[0], history) == "failure"

"""call_jenkins_language must hit the unified gated pipeline and fail loudly.

Regression for PR #409 (llada-8b-base): the trigger pointed at the retired
bash pipeline `core/job/score_plugins`, whose relative `cd language` breaks
under the current image's `WORKDIR /language`. Every job died in seconds and
the helper swallowed any HTTP error, so the workflow reported success.
"""
import os
from unittest import mock

import pytest
import requests

from brainscore_language.submission.jenkins import call_jenkins_language

PLUGIN_INFO = {
    "domain": "language",
    "new_models": "llada-8b-base",
    "new_benchmarks": "",
    "email": "submitter@example.edu",
    "user_id": "844",
    "public": True,
    "run_score": "True",
}

ENV = {"JENKINS_USER": "u", "JENKINS_TOKEN": "t", "JENKINS_TRIGGER": "trigger-secret"}


def _response(status):
    r = requests.Response()
    r.status_code = status
    return r


@mock.patch.dict(os.environ, ENV)
def test_targets_the_gated_pipeline():
    with mock.patch("requests.get", return_value=_response(201)) as get:
        call_jenkins_language(PLUGIN_INFO)
    url = get.call_args.args[0]
    assert "/job/core/job/gated_score_plugins/buildWithParameters" in url
    assert "score_plugins/buildWithParameters" not in url.replace("gated_score_plugins", "")


@mock.patch.dict(os.environ, ENV)
def test_forwards_submitter_identity():
    with mock.patch("requests.get", return_value=_response(201)) as get:
        call_jenkins_language(PLUGIN_INFO)
    params = get.call_args.kwargs["params"]
    assert params["email"] == "submitter@example.edu"
    assert params["user_id"] == "844"
    assert params["new_models"] == "llada-8b-base"
    assert "new_benchmarks" not in params  # empty values are dropped


@mock.patch.dict(os.environ, ENV)
def test_accepts_json_string():
    import json
    with mock.patch("requests.get", return_value=_response(201)) as get:
        call_jenkins_language(json.dumps(PLUGIN_INFO))
    assert get.call_args.kwargs["params"]["domain"] == "language"


@mock.patch.dict(os.environ, ENV)
@pytest.mark.parametrize("status", [401, 403, 404, 500])
def test_raises_on_rejection(status):
    with mock.patch("requests.get", return_value=_response(status)):
        with pytest.raises(RuntimeError, match=f"HTTP {status}"):
            call_jenkins_language(PLUGIN_INFO)


@mock.patch.dict(os.environ, ENV)
def test_error_does_not_leak_token_or_email():
    with mock.patch("requests.get", return_value=_response(403)):
        with pytest.raises(RuntimeError) as exc:
            call_jenkins_language(PLUGIN_INFO)
    msg = str(exc.value)
    assert "trigger-secret" not in msg
    assert "submitter@example.edu" not in msg
    assert "buildWithParameters" not in msg


@mock.patch.dict(os.environ, ENV)
def test_connection_errors_propagate():
    with mock.patch("requests.get", side_effect=requests.ConnectionError("down")):
        with pytest.raises(requests.ConnectionError):
            call_jenkins_language(PLUGIN_INFO)

"""Jenkins trigger for language scoring.

Kept free of any brainscore_language.submission.endpoints import: that module
resolves the database secret at import time, which is neither available nor
wanted on the GitHub Actions runner that fires the trigger.
"""
import json
import os
from typing import Dict, List, Union

import requests
from requests.auth import HTTPBasicAuth


def call_jenkins_language(plugin_info: Union[str, Dict[str, Union[List[str], str]]]):
    """Trigger the unified gated scoring pipeline for a language plugin.

    Raises on any non-2xx response so the calling workflow step fails instead
    of reporting a kickoff that Jenkins never accepted.
    """
    jenkins_base = "http://www.brain-score-jenkins.com:8080"
    jenkins_user = os.environ['JENKINS_USER']
    jenkins_token = os.environ['JENKINS_TOKEN']
    jenkins_trigger = os.environ['JENKINS_TRIGGER']
    jenkins_job = "core/job/gated_score_plugins"

    url = f'{jenkins_base}/job/{jenkins_job}/buildWithParameters?token={jenkins_trigger}'

    if isinstance(plugin_info, str):
        plugin_info = json.loads(plugin_info)

    payload = {k: v for k, v in plugin_info.items() if plugin_info[k]}
    auth_basic = HTTPBasicAuth(username=jenkins_user, password=jenkins_token)
    response = requests.get(url, params=payload, auth=auth_basic)
    # Do not include the URL or payload in the error: the query string carries
    # the trigger token and the submitter's email.
    if not response.ok:
        raise RuntimeError(f'Jenkins rejected the scoring trigger: HTTP {response.status_code}')
    return response

"""
GitHub Actions helper functions for language domain
Supports the plugin submission prepare and validate workflows

Functions:
- validate_pr: Validate PR for automerge eligibility
- trigger_layer_mapping: Trigger Jenkins layer mapping job
- send_failure_email: Send failure notification email
"""

import json
import os
import requests
import sys
import smtplib
import argparse
import time
from typing import Union
from email.mime.text import MIMEText

BASE_URL = "https://api.github.com/repos/brain-score/language"


def get_data(url: str, token: str = None, accept_header: str = None) -> dict:
    """Fetch data from GitHub API"""
    headers = {}
    if token:
        headers["Authorization"] = f"token {token}"
    if accept_header:
        headers["Accept"] = accept_header
    
    r = requests.get(url, headers=headers)
    assert r.status_code == 200, f'{r.status_code}: {r.reason}'
    return r.json()


def get_check_run_result(context: str, check_runs_json: dict) -> Union[str, None]:
    """Get the latest check run result for a given context (check run name)"""
    # Check runs API returns a dict with 'check_runs' key
    check_runs = check_runs_json.get('check_runs', [])
    
    # Find check runs matching the context (by name) 
    matching_runs = [
        {
            'updated_at': run.get('updated_at', run.get('completed_at', '')),
            'status': run.get('status'),
            'conclusion': run.get('conclusion')
        }
        for run in check_runs if run.get('name') == context
    ]
    
    if not matching_runs:
        # If no check runs exist yet, treat as pending (not null)
        return "pending"
    
    # Get the most recent run
    last_run = max(matching_runs, key=lambda x: x['updated_at'])
    
    # Map status and conclusion to result
    status = last_run['status']
    conclusion = last_run['conclusion']
    
    if conclusion == "success":
        return "success"
    elif conclusion == "failure":
        return "failure"
    elif status == "in_progress" or conclusion is None:
        return "pending"
    else:
        # Handle other states (queued, etc.) as pending
        return "pending"


def validate_pr(pr_number: int, pr_head: str, is_automerge_web: bool, token: str, 
                poll_interval: int = 30, max_wait_time: int = 7200) -> dict:
    """
    Validate PR for automerge eligibility
    
    Polls test status every poll_interval seconds until all tests are complete
    (success or failure), or max_wait_time is reached.
    
    Args:
        pr_number: PR number
        pr_head: PR head commit SHA
        is_automerge_web: Whether this is an automerge-web PR
        token: GitHub token
        poll_interval: Seconds to wait between polls (default: 30)
        max_wait_time: Maximum seconds to wait for tests (default: 7200 = 2 hours)
    
    Returns:
        dict with keys:
        - is_automergeable: bool
        - all_tests_pass: bool
        - test_results: dict mapping test context to result
    """
    # Check required test contexts (adjust these for language domain)
    required_contexts = [
        "Language Unittests, Plugins",
        "Language Unittests, Non-Plugins",
        "Language Integration Tests",
        "docs/readthedocs.org:brain-score-language"
    ]
    
    RTD_CONTEXT = "docs/readthedocs.org:brain-score-language"
    RTD_NULL_THRESHOLD = 4  # Number of consecutive nulls before ignoring
    RTD_TIMEOUT_SECONDS = 120  # 2 minutes
    
    start_time = time.time()
    test_results = {}
    rtd_null_count = 0
    ignore_rtd = False
    
    # Fetch PR to get merge_commit_sha (for fork PRs, check runs are attached to merge commit)
    pr_url = f"{BASE_URL}/pulls/{pr_number}"
    pr_data = get_data(pr_url, token)
    merge_commit_sha = pr_data.get('merge_commit_sha')
    
    # Use merge_commit_sha if available, otherwise fall back to pr_head
    commit_sha_for_checks = merge_commit_sha if merge_commit_sha else pr_head
    
    while True:
        # Get check runs using Check Runs API (works for fork PRs)
        check_runs_url = f"{BASE_URL}/commits/{commit_sha_for_checks}/check-runs"
        check_runs_json = get_data(check_runs_url, token, accept_header="application/vnd.github+json")
        
        # Check each required context
        test_results = {}
        has_pending = False
        
        elapsed = time.time() - start_time
        
        # Check if RTD check run exists (for null detection)
        check_runs = check_runs_json.get('check_runs', [])
        rtd_check_run_exists = any(run.get('name') == RTD_CONTEXT for run in check_runs)
        
        for context in required_contexts:
            result = get_check_run_result(context, check_runs_json)
            test_results[context] = result
            
            # Special handling for ReadTheDocs: track consecutive nulls within 2 minutes
            if context == RTD_CONTEXT:
                # Check if RTD check run doesn't exist (treat as null)
                if not rtd_check_run_exists:
                    # Only track nulls if we're still within the 2-minute window
                    if elapsed <= RTD_TIMEOUT_SECONDS:
                        rtd_null_count += 1
                        print(f"ReadTheDocs check returned null (consecutive null count: {rtd_null_count}, elapsed: {elapsed:.1f}s)", file=sys.stderr)
                        
                        # If we hit the threshold, ignore RTD
                        if rtd_null_count >= RTD_NULL_THRESHOLD and not ignore_rtd:
                            ignore_rtd = True
                            print(f"ReadTheDocs has returned null {RTD_NULL_THRESHOLD} consecutive times within {elapsed:.1f}s. Ignoring RTD check.", file=sys.stderr)
                    else:
                        # We're past the 2-minute window, don't track nulls anymore
                        if not ignore_rtd:
                            print(f"ReadTheDocs check returned null but we're past {RTD_TIMEOUT_SECONDS}s timeout. Will not ignore RTD.", file=sys.stderr)
                else:
                    # RTD check run exists, reset counter (only if we haven't already ignored it)
                    if rtd_null_count > 0 and not ignore_rtd:
                        print(f"ReadTheDocs check returned non-null result ({result}), resetting null counter", file=sys.stderr)
                    if not ignore_rtd:
                        rtd_null_count = 0
            
            # Check if any test is still pending (skip RTD if we're ignoring it)
            if context == RTD_CONTEXT and ignore_rtd:
                # Skip RTD from pending check if we're ignoring it
                continue
            elif result is None or result == "pending":
                has_pending = True
        
        # If no pending tests, we're done
        if not has_pending:
            break
        
        # Check if we've exceeded max wait time
        elapsed = time.time() - start_time
        if elapsed >= max_wait_time:
            print(f"Warning: Max wait time ({max_wait_time}s) reached. Some tests still pending.", file=sys.stderr)
            break
        
        # Wait before next poll
        print(f"Tests still pending. Waiting {poll_interval}s before next check...", file=sys.stderr)
        print(f"Current status: {json.dumps(test_results)}", file=sys.stderr)
        time.sleep(poll_interval)
    
    # Determine if all tests pass (exclude RTD if we're ignoring it)
    tests_to_check = test_results.copy()
    if ignore_rtd:
        print(f"Excluding ReadTheDocs from validation check (was null {RTD_NULL_THRESHOLD} consecutive times)", file=sys.stderr)
        tests_to_check.pop(RTD_CONTEXT, None)
    
    all_tests_pass = all(
        result == "success" for result in tests_to_check.values() if result is not None
    )
    
    # Debug: Print final test statuses before validation results
    print("Final test statuses before validation:", file=sys.stderr)
    print(json.dumps(test_results, indent=2), file=sys.stderr)
    print(f"All tests pass: {all_tests_pass}", file=sys.stderr)
    
    # Check if PR is automergeable
    # (PR must have submission_prepared label and all tests must pass)
    labels_url = f"{BASE_URL}/issues/{pr_number}/labels"
    labels_json = get_data(labels_url, token)
    label_names = [label['name'] for label in labels_json]
    has_submission_prepared_label = any(
        label['name'] == 'submission_prepared'
        for label in labels_json
    )
    
    # Debug: Print label information
    print(f"PR labels: {label_names}", file=sys.stderr)
    print(f"Has submission_prepared label: {has_submission_prepared_label}", file=sys.stderr)
    
    is_automergeable = has_submission_prepared_label and all_tests_pass
    
    # Debug: Print final determination
    print(f"Is automergeable: {is_automergeable} (has_submission_prepared_label={has_submission_prepared_label}, all_tests_pass={all_tests_pass})", file=sys.stderr)
    
    return {
        "is_automergeable": is_automergeable,
        "all_tests_pass": all_tests_pass,
        "test_results": test_results
    }


def trigger_update_existing_metadata(plugin_dirs: str, plugin_type: str, domain: str,
                                     jenkins_user: str, jenkins_token: str, jenkins_trigger: str,
                                     metadata_and_layer_map: dict = None):
    """Trigger Jenkins update_existing_metadata job"""
    import json
    
    # Build Jenkins trigger URL
    jenkins_base = "http://www.brain-score-jenkins.com:8080"
    url = f"{jenkins_base}/job/update_existing_metadata/buildWithParameters?token={jenkins_trigger}"
    
    # Prepare payload
    payload = {
        "domain": domain,
        "plugin_dirs": plugin_dirs,
        "plugin_type": plugin_type,
        "update_metadata_only": "true"
    }
    
    # Add metadata_and_layer_map if provided (JSON-serialize nested dict)
    if metadata_and_layer_map:
        payload["metadata_and_layer_map"] = json.dumps(metadata_and_layer_map)
    
    # Trigger Jenkins
    from requests.auth import HTTPBasicAuth
    auth = HTTPBasicAuth(username=jenkins_user, password=jenkins_token)
    
    try:
        response = requests.get(url, params=payload, auth=auth)
        response.raise_for_status()
        print(f"Successfully triggered update_existing_metadata for {plugin_type}: {plugin_dirs}")
    except Exception as e:
        print(f"Failed to trigger Jenkins update_existing_metadata: {e}")
        raise


def trigger_layer_mapping(new_models: str, pr_number: int, source_repo: str, 
                         source_branch: str, jenkins_user: str, jenkins_user_api: str,
                         jenkins_token: str, jenkins_trigger: str):
    """Trigger Jenkins layer mapping job"""
    import json
    
    # Parse new_models (should be JSON array string)
    try:
        models_list = json.loads(new_models) if new_models else []
    except json.JSONDecodeError:
        models_list = new_models.split(',') if new_models else []
    
    if not models_list or models_list == []:
        print("No new models to map, skipping layer mapping")
        return
    
    # Build Jenkins trigger URL
    jenkins_base = "http://www.brain-score-jenkins.com:8080"
    url = f"{jenkins_base}/job/{jenkins_trigger}/buildWithParameters"
    
    # Prepare payload
    payload = {
        "NEW_MODELS": ",".join(models_list),
        "PR_NUMBER": str(pr_number),
        "SOURCE_REPO": source_repo,
        "SOURCE_BRANCH": source_branch,
        "TOKEN": jenkins_token
    }
    
    # Trigger Jenkins
    from requests.auth import HTTPBasicAuth
    auth = HTTPBasicAuth(username=jenkins_user, password=jenkins_user_api)
    
    try:
        response = requests.post(url, params=payload, auth=auth)
        response.raise_for_status()
        print(f"Successfully triggered layer mapping for models: {models_list}")
    except Exception as e:
        print(f"Failed to trigger Jenkins layer mapping: {e}")
        raise


def send_failure_email(email: str, pr_number: str, failure_reason: str,
                       mail_username: str, mail_password: str):
    """Send failure notification email to user"""
    subject = "Brain-Score Language Submission Failed"
    body = f"""Your Brain-Score language submission did not pass checks.

Failure reason: {failure_reason}

Please review the test results and update the PR at:
https://github.com/brain-score/language/pull/{pr_number}

Or send in an updated submission via the website.
"""
    
    msg = MIMEText(body)
    msg['Subject'] = subject
    msg['From'] = "Brain-Score"
    msg['To'] = email

    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp_server:
            smtp_server.login(mail_username, mail_password)
            smtp_server.sendmail(mail_username, email, msg.as_string())
        print(f"Email sent to {email}")
    except Exception as e:
        print(f"Failed to send email: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(description='GitHub Actions helper for language domain')
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Validate PR command
    validate_parser = subparsers.add_parser('validate_pr', help='Validate PR for automerge')
    validate_parser.add_argument('--pr-number', type=int, required=True)
    validate_parser.add_argument('--pr-head', type=str, required=True)
    validate_parser.add_argument('--is-automerge-web', type=str, default='false')
    validate_parser.add_argument('--token', type=str, default=os.getenv('GITHUB_TOKEN'))
    
    # Trigger update existing metadata command
    update_metadata_parser = subparsers.add_parser('trigger_update_existing_metadata', help='Trigger update existing metadata')
    update_metadata_parser.add_argument('--plugin-dirs', type=str, required=True)
    update_metadata_parser.add_argument('--plugin-type', type=str, required=True)
    update_metadata_parser.add_argument('--domain', type=str, default='language')
    update_metadata_parser.add_argument('--metadata-and-layer-map-b64', type=str, default='')
    
    # Trigger layer mapping command
    mapping_parser = subparsers.add_parser('trigger_layer_mapping', help='Trigger layer mapping')
    mapping_parser.add_argument('--new-models', type=str, required=True)
    mapping_parser.add_argument('--pr-number', type=int, required=True)
    mapping_parser.add_argument('--source-repo', type=str, required=True)
    mapping_parser.add_argument('--source-branch', type=str, required=True)
    
    # Extract email command
    extract_parser = subparsers.add_parser('extract_email', help='Extract user email')
    extract_parser.add_argument('--pr-username', type=str, required=True)
    extract_parser.add_argument('--pr-title', type=str, default='')
    extract_parser.add_argument('--is-automerge-web', type=str, default='false')
    
    # Send failure email command
    email_parser = subparsers.add_parser('send_failure_email', help='Send failure email')
    email_parser.add_argument('email', type=str)
    email_parser.add_argument('pr_number', type=str)
    email_parser.add_argument('failure_reason', type=str)
    email_parser.add_argument('mail_username', type=str)
    email_parser.add_argument('mail_password', type=str)
    
    args = parser.parse_args()
    
    if args.command == 'extract_email':
        from brainscore_core.submission.database import email_from_uid
        from brainscore_core.submission.endpoints import UserManager
        
        is_automerge_web = args.is_automerge_web.lower() == 'true'
        
        if is_automerge_web:
            # Extract user ID from PR title
            import re
            match = re.search(r'\(user:([^)]+)\)', args.pr_title)
            if match:
                bs_uid = match.group(1)
                db_secret = os.getenv('BSC_DATABASESECRET')
                user_manager = UserManager(db_secret=db_secret)
                email = email_from_uid(int(bs_uid))
                if not email:
                    # Fallback to default email if database lookup returns no email
                    email = "mferg@mit.edu"
                    print(f"Could not find email in database for user {bs_uid}, using default: {email}", file=sys.stderr)
            else:
                print("Could not extract user ID from PR title", file=sys.stderr)
                sys.exit(1)
        else:
            # Get email from GitHub username using GitHub API
            token = os.getenv('GITHUB_TOKEN')
            headers = {"Authorization": f"token {token}"} if token else {}
            user_url = f"https://api.github.com/users/{args.pr_username}"
            user_data = get_data(user_url, token)
            email = user_data.get('email')
            if not email:
                # Try to get from events
                events_url = f"https://api.github.com/users/{args.pr_username}/events/public"
                events = get_data(events_url, token)
                for event in events[:10]:  # Check recent events
                    if 'payload' in event and 'commits' in event['payload']:
                        for commit in event['payload']['commits']:
                            if 'author' in commit and 'email' in commit['author']:
                                email = commit['author']['email']
                                break
                    if email:
                        break
            if not email:
                # Fallback to default email if real email not found
                email = "mferg@mit.edu"
                print(f"Could not find email for user, using default Brain-Score submission (mferg): {email}", file=sys.stderr)
        
        print(email)
        
    elif args.command == 'validate_pr':
        is_automerge_web = args.is_automerge_web.lower() == 'true'
        result = validate_pr(args.pr_number, args.pr_head, is_automerge_web, args.token)
        print(json.dumps(result))
        
    elif args.command == 'trigger_update_existing_metadata':
        # Decode metadata_and_layer_map if provided
        metadata_and_layer_map = None
        if args.metadata_and_layer_map_b64:
            import base64
            try:
                metadata_json = base64.b64decode(args.metadata_and_layer_map_b64).decode('utf-8')
                metadata_and_layer_map = json.loads(metadata_json)
            except Exception as e:
                print(f"Warning: Failed to decode metadata_and_layer_map: {e}", file=sys.stderr)
        
        trigger_update_existing_metadata(
            plugin_dirs=args.plugin_dirs,
            plugin_type=args.plugin_type,
            domain=args.domain,
            metadata_and_layer_map=metadata_and_layer_map,
            jenkins_user=os.getenv('JENKINS_USER'),
            jenkins_token=os.getenv('JENKINS_TOKEN'),
            jenkins_trigger=os.getenv('JENKINS_TRIGGER')
        )
        
    elif args.command == 'trigger_layer_mapping':
        trigger_layer_mapping(
            new_models=args.new_models,
            pr_number=args.pr_number,
            source_repo=args.source_repo,
            source_branch=args.source_branch,
            jenkins_user=os.getenv('JENKINS_USER'),
            jenkins_user_api=os.getenv('JENKINS_USER_API'),
            jenkins_token=os.getenv('JENKINS_TOKEN'),
            jenkins_trigger=os.getenv('JENKINS_TRIGGER')
        )
        
    elif args.command == 'send_failure_email':
        send_failure_email(
            email=args.email,
            pr_number=args.pr_number,
            failure_reason=args.failure_reason,
            mail_username=args.mail_username,
            mail_password=args.mail_password
        )
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()

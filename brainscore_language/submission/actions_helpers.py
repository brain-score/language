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


def get_data(url: str, token: str = None) -> dict:
    """Fetch data from GitHub API"""
    headers = {}
    if token:
        headers["Authorization"] = f"token {token}"
    
    r = requests.get(url, headers=headers)
    assert r.status_code == 200, f'{r.status_code}: {r.reason}'
    return r.json()


def get_statuses_result(context: str, statuses_json: dict) -> Union[str, None]:
    """Get the latest status result for a given context"""
    statuses = [
        {'end_time': status['updated_at'], 'result': status['state']}
        for status in statuses_json if status['context'] == context
    ]
    
    if not statuses:
        return None
    
    last_status = max(statuses, key=lambda x: x['end_time'])
    return last_status['result']


def validate_pr(pr_number: int, pr_head: str, is_automerge_web: bool, token: str,
                poll_interval: int = 30, max_wait_time: int = 7200) -> dict:
    """Require every test status on the submitted head, including after timeout.

    This checks results reported by the test runners; it does not execute tests.
    Missing statuses and pending results never count as a successful validation.
    """
    required_contexts = [
        "Language Unittests, Plugins",
        "Language Unittests, Non-Plugins",
        "Language Integration Tests",
        "docs/readthedocs.org:brain-score-language",
    ]
    start_time = time.monotonic()
    while True:
        # Combined status returns the current status per context, rather than
        # a page of historical updates that may omit a required context.
        status_url = f"{BASE_URL}/commits/{pr_head}/status?per_page=100"
        statuses = get_data(status_url, token)["statuses"]
        test_results = {
            context: get_statuses_result(context, statuses)
            for context in required_contexts
        }
        print(f"Current test statuses: {json.dumps(test_results)}", file=sys.stderr)
        if all(result == "success" for result in test_results.values()):
            break
        if any(result not in (None, "pending", "success") for result in test_results.values()):
            break
        elapsed = time.monotonic() - start_time
        if elapsed >= max_wait_time:
            print("Validation timed out with missing or pending required statuses", file=sys.stderr)
            break
        time.sleep(min(poll_interval, max_wait_time - elapsed))

    # A new push can arrive while we wait. Never prepare or approve the new
    # revision using results from the previous submitted head.
    pr = get_data(f"{BASE_URL}/pulls/{pr_number}", token)
    head_matches = pr["head"]["sha"] == pr_head
    all_tests_pass = head_matches and all(
        test_results[context] == "success" for context in required_contexts
    )
    if not head_matches:
        print("PR head changed during validation; results belong to an older revision", file=sys.stderr)
    has_submission_prepared_label = any(
        label["name"] == "submission_prepared" for label in pr["labels"]
    )
    return {
        "is_automergeable": has_submission_prepared_label and all_tests_pass,
        "all_tests_pass": all_tests_pass,
        "head_matches": head_matches,
        "test_results": test_results,
    }


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
    validate_parser.add_argument('--poll-interval', type=int, default=30)
    validate_parser.add_argument('--max-wait-time', type=int, default=7200)
    
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
        result = validate_pr(args.pr_number, args.pr_head, is_automerge_web, args.token,
                             poll_interval=args.poll_interval, max_wait_time=args.max_wait_time)
        print(json.dumps(result))
        
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

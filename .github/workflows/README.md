# Language Domain Workflow Orchestration

## Overview

This directory contains a unified workflow for LLM evaluations in the Brain-Score language domain. The system uses a single orchestrated workflow (`plugin_submission_orchestrator.yml`) that handles all steps from change detection through post-merge scoring.

## Architecture

### Unified Workflow Design

A single workflow handles all phases of the submission lifecycle:

- **`plugin_submission_orchestrator.yml`** — The main orchestrator. Detects changes, validates PRs, generates metadata, auto-merges (web submissions only), triggers Jenkins scoring, and sends failure notifications.

### Reusable Workflows

- **`metadata_handler.yml`** — Processes plugin metadata (can be triggered standalone via `workflow_dispatch`)
- **`user_notification_handler.yml`** — Handles email extraction and notifications

### Helper Scripts

- **`brainscore_language/submission/actions_helpers.py`** — Python CLI used by the orchestrator for:
  - `validate_pr` — Polls GitHub commit statuses API to check if tests pass
  - `trigger_update_existing_metadata` — Triggers the Jenkins `update_existing_metadata` job
  - `trigger_layer_mapping` — Triggers the Jenkins layer mapping job (non-language domains only)
  - `extract_email` — Resolves submitter email from GitHub username or Brain-Score user ID
  - `send_failure_email` — Sends failure notification emails via Gmail SMTP
- **`brainscore_language/submission/endpoints.py`** — Defines `call_jenkins_language()` which triggers the `core/job/score_plugins` Jenkins job with CSRF crumb handling
- **`brainscore_language/submission/hardcoded_metadata.py`** — Generates `metadata.yml` for plugins that don't have one
- **`brainscore_language/submission/config.py`** — Provides `get_database_secret()` from environment

## Orchestrator Jobs

The orchestrator has 7 numbered jobs:

| # | Job | When it runs |
|---|-----|-------------|
| 1 | **Detect Changes** | Every `pull_request` and `pull_request_target` (merged) event |
| 2 | **Validate PR** | Plugin PRs only (skipped for metadata-only PRs) |
| 3 | **Handle Metadata-Only PR** | Only when PR contains only `metadata.yml` changes |
| 4 | **Generate Mutations and Commit** | When metadata generation or layer mapping is needed and tests pass |
| 5 | **Auto-merge** | Web submissions only, when tests pass and `submission_prepared` label exists |
| 6 | **Post-Merge Kickoff** | After PR is merged (`pull_request_target` with `merged == true`) |
| 7 | **Notify on Failure** | When any upstream job fails or tests don't pass |

### Job Details

**1. Detect Changes** — Runs `brainscore_core.plugin_management.parse_plugin_changes.get_scoring_info()` on the changed files to determine:
- `has_plugins` — Whether plugin files were changed
- `plugin_type` — `models`, `benchmarks`, or `models,benchmarks`
- `plugin_dirs` — Comma-separated paths (e.g. `brainscore_language/models/mymodel`)
- `metadata_only` — Whether only `metadata.yml` files were changed
- `needs_metadata_generation` — Whether any plugin directory is missing `metadata.yml`
- `needs_mapping` — Whether layer mapping is needed (always `false` for language domain)
- `needs_scoring` — Whether scoring should be triggered post-merge
- `is_automergeable` — From `get_scoring_info` output
- `has_new_models` — Whether new model plugins were detected

**2. Validate PR** — Polls the GitHub commit statuses API (`/commits/{sha}/statuses`) to check test results for:
- `Language Unittests, Plugins`
- `Language Unittests, Non-Plugins`
- `Language Integration Tests`
- `docs/readthedocs.org:brain-score-language` (with special RTD null-handling: if RTD returns null 4+ consecutive times within 2 minutes, it's excluded from validation)

If metadata already exists and tests pass, adds the `submission_prepared` label.

**3. Handle Metadata-Only PR** — Adds the `only_update_metadata` label, which triggers a new orchestrator run.

**4. Generate Mutations and Commit** — Runs `hardcoded_metadata.py` to generate `metadata.yml` for plugins missing it. Commits and pushes via PAT (to trigger a new workflow run). Adds the `submission_prepared` label. Uses `always()` in its condition to run even when `validate_pr` is skipped.

**5. Auto-merge** — Only merges **web submissions** (PR title contains `(user:<id>)`). Non-web submissions are never auto-merged. For plugin PRs, requires `submission_prepared` label + tests passing. For metadata-only PRs, checks tests directly. Uses `hmarr/auto-approve-action` for approval and `plm9606/automerge_actions` for squash merge.

**6. Post-Merge Kickoff** — Triggered by `pull_request_target` (merged). For plugin PRs: extracts submitter email (encrypted/decrypted with `EMAIL_ENCRYPTION_KEY`), builds plugin info JSON, and calls `call_jenkins_language()` to trigger the `core/job/score_plugins` Jenkins job. For metadata-only PRs: triggers `update_existing_metadata` Jenkins job.

**7. Notify on Failure** — Sends an email to the submitter when any job fails or tests don't pass. Extracts email using the same web/non-web logic as post-merge. Falls back to `mferg@mit.edu` if email lookup fails.

## Workflow Flows

### Web Plugin PR (needs metadata generation) — 3 runs

```
Run 1 (PR opened):
Plugin Submission Orchestrator
├─ 1. Detect Changes ✓
│   └─ has_plugins=true, needs_metadata_generation=true
├─ 2. Validate PR ✓
│   └─ Polls tests, tests pass, but no submission_prepared label yet
├─ 3. Handle Metadata-Only PR (skipped — not metadata-only)
├─ 4. Generate Mutations and Commit ✓
│   ├─ Generates metadata.yml via hardcoded_metadata.py
│   ├─ Commits + pushes to PR branch (triggers new run)
│   └─ Adds submission_prepared label
├─ 5. Auto-merge (skipped — tests must re-run after new commit)
└─ (new commit triggers Run 2)

Run 2 (synchronize event from commit push):
Plugin Submission Orchestrator
├─ 1. Detect Changes ✓
│   └─ has_plugins=true, needs_metadata_generation=false (metadata exists now)
├─ 2. Validate PR ✓
│   └─ Polls tests, tests pass, adds submission_prepared (already exists)
├─ 3. Handle Metadata-Only PR (skipped)
├─ 4. Generate Mutations and Commit (skipped — no mutations needed)
├─ 5. Auto-merge ✓ (web submission, submission_prepared + tests pass)
│   ├─ Approves PR
│   └─ Squash merges to main
└─ (merge triggers Run 3)

Run 3 (pull_request_target — merged):
Plugin Submission Orchestrator
├─ 1. Detect Changes ✓
├─ 6. Post-Merge Kickoff ✓
│   ├─ Extracts email (from user ID in PR title → database lookup)
│   ├─ Builds plugin info JSON
│   └─ Triggers Jenkins core/job/score_plugins
└─ 7. Notify on Failure (skipped — no failures)
```

### Web Plugin PR (metadata already present) — 2 runs

```
Run 1 (PR opened):
Plugin Submission Orchestrator
├─ 1. Detect Changes ✓
│   └─ has_plugins=true, needs_metadata_generation=false
├─ 2. Validate PR ✓
│   └─ Tests pass, adds submission_prepared label
├─ 3. Handle Metadata-Only PR (skipped)
├─ 4. Generate Mutations and Commit (skipped — no mutations needed)
├─ 5. Auto-merge ✓
│   └─ Squash merges
└─ (merge triggers Run 2)

Run 2 (merged):
Plugin Submission Orchestrator
├─ 1. Detect Changes ✓
├─ 6. Post-Merge Kickoff ✓
│   └─ Triggers Jenkins scoring
└─ 7. Notify on Failure (skipped)
```

### Non-Web Plugin PR — no auto-merge

```
Run 1 (PR opened):
Plugin Submission Orchestrator
├─ 1. Detect Changes ✓
├─ 2. Validate PR ✓
│   └─ Tests pass (or fail)
├─ 4. Generate Mutations and Commit (if metadata needed + tests pass)
├─ 5. Auto-merge (skipped — non-web submission, automerge disabled)
└─ (manual merge required)

After manual merge:
Plugin Submission Orchestrator
├─ 1. Detect Changes ✓
├─ 6. Post-Merge Kickoff ✓
│   └─ Triggers Jenkins scoring
└─ 7. Notify on Failure (skipped)
```

### Metadata-Only PR (web submission) — 2 runs

```
Run 1:
Plugin Submission Orchestrator
├─ 1. Detect Changes ✓ (metadata_only=true)
├─ 2. Validate PR (skipped — metadata-only)
├─ 3. Handle Metadata-Only PR ✓
│   └─ Adds only_update_metadata label (triggers new run)
└─ (label triggers Run 2)

Run 2 (labeled event):
Plugin Submission Orchestrator
├─ 1. Detect Changes ✓ (metadata_only=true)
├─ 2. Validate PR (skipped)
├─ 3. Handle Metadata-Only PR ✓ (label already exists, exits)
├─ 5. Auto-merge ✓ (web submission, checks tests directly for metadata-only)
│   └─ Squash merges
├─ 6. Post-Merge Kickoff ✓
│   └─ Triggers update_existing_metadata Jenkins job
└─ 7. Notify on Failure (skipped)
```

### Benchmark-Only PR

Benchmark-only PRs have `needs_scoring=false` and `needs_metadata_generation=false`. They follow the same flow as plugin PRs but skip scoring in post-merge.

### Test Failure Flow

```
Plugin Submission Orchestrator
├─ 1. Detect Changes ✓
├─ 2. Validate PR ✓ (job succeeds, but all_tests_pass=false)
├─ 4. Generate Mutations (skipped — tests didn't pass)
├─ 5. Auto-merge (skipped — tests didn't pass)
└─ 7. Notify on Failure ✓
    └─ Sends email with failed test names
```

## Security & Masking

- **Email encryption**: Emails are encrypted with AES-256-CBC using `EMAIL_ENCRYPTION_KEY` and decrypted only when needed. `::add-mask::` is used to prevent log exposure.
- **Jenkins credentials**: Token passed as query parameter, basic auth for API calls. All Jenkins output is suppressed (`>/dev/null 2>&1`).
- **PAT usage**: `GH_MFERG_PAT` is used for push operations (to trigger new workflow runs) and PR operations. Git push uses `>/dev/null 2>&1` to suppress token in error output.
- **Sensitive data in logs**: Plugin info is logged with email masked. Jenkins URLs are masked by GitHub Actions secret masking.

## Configuration

### Required Secrets

| Secret | Purpose |
|--------|---------|
| `BSC_DATABASESECRET` | Database connection secret |
| `AWS_ACCESS_KEY_ID` | AWS credentials |
| `AWS_SECRET_ACCESS_KEY` | AWS credentials |
| `JENKINS_USER` | Jenkins username for scoring jobs |
| `JENKINS_TOKEN` | Jenkins API token for scoring jobs |
| `JENKINS_TRIGGER` | Jenkins job trigger token |
| `JENKINS_MAPPING_USER` | Jenkins username for layer mapping (non-language) |
| `JENKINS_MAPPING_USER_API` | Jenkins API key for layer mapping |
| `JENKINS_MAPPING_TOKEN` | Jenkins token for layer mapping |
| `JENKINS_MAPPING_URL` | Jenkins URL for layer mapping |
| `GH_MFERG_PAT` | GitHub PAT for push/PR operations |
| `GITHUB_TOKEN` | GitHub API token (auto-provided) |
| `GMAIL_USERNAME` | Gmail account for failure notifications |
| `GMAIL_PASSWORD` | Gmail app password for notifications |
| `EMAIL_ENCRYPTION_KEY` | AES-256-CBC key for email encryption |

### Environment Variables (set in workflow)

| Variable | Value | Purpose |
|----------|-------|---------|
| `DOMAIN` | `language` | Brain-Score domain identifier |
| `DOMAIN_ROOT` | `brainscore_language` | Root package directory |
| `PYTHON_VERSION` | `3.11` | Python version for all jobs |

### Triggers

```yaml
on:
  pull_request:
    types: [opened, synchronize, labeled, reopened]
  pull_request_target:
    types: [closed]
    branches: [main]
```

- `pull_request` events run Steps 1–5 (pre-merge)
- `pull_request_target` (closed + merged) runs Steps 1 + 6 (post-merge)

### Permissions

```yaml
permissions:
  contents: write        # Push commits, merge PRs
  pull-requests: write   # Add labels, approve PRs
  issues: write          # Label management
  checks: read           # Read check run status
  statuses: read         # Read commit statuses
```

## Jenkins Integration

### Scoring (`core/job/score_plugins`)

Triggered by `call_jenkins_language()` in `endpoints.py`. Sends parameters:
- `new_models`, `domain`, `email`, `plugin_dirs`, `plugin_type`
- `competition`, `model_type`, `public`, `user_id`

The function uses CSRF crumb handling (fetches crumb from `/crumbIssuer/api/json`, includes in POST headers) and basic auth.

### Metadata Update (`update_existing_metadata`)

Triggered by `actions_helpers.py trigger_update_existing_metadata` for metadata-only PRs. Sends `domain`, `plugin_dirs`, `plugin_type`, and serialized metadata JSON.

### Layer Mapping (non-language domains only)

Triggered by `actions_helpers.py trigger_layer_mapping`. Always skipped for language domain (`needs_mapping` is always `false` when `DOMAIN == 'language'`).

## Debugging

1. Check workflow logs in GitHub Actions — each job is numbered for easy identification
2. Review job outputs in the "Detect Changes" step for classification details
3. Check Jenkins at `brain-score-jenkins.com:8080` for scoring job status
4. Review email notifications for failures
5. For test polling issues, check the `validate_pr` stderr output for RTD null counts and polling status

## Comparison to Vision

| Aspect | Vision (Old) | Language (New) |
|--------|-------------|----------------|
| **Workflows** | 10+ separate files | 1 unified workflow + 2 reusable |
| **Architecture** | Multiple workflows with complex triggers | Single orchestrator |
| **Entry Point** | Multiple triggers | Single entry point |
| **Dependencies** | Implicit, complex | Explicit, sequential with numbered jobs |
| **Visibility** | Hard to see flow | Clear numbered steps in Actions UI |
| **Error Handling** | Scattered | Centralized failure notification job |
| **Auto-merge** | All submissions | Web submissions only |
| **Jenkins Job** | `dev_score_plugins` | `core/job/score_plugins` |

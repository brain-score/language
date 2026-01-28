# Language Domain Workflow Orchestration

## Overview

This directory contains two complementary workflows for LLM evaluations in the Brain-Score language domain. The system uses a two-phase approach: prepare (commits changes to PR) and validate (handles downstream steps after PR is complete).

## Architecture

### Two-Phase Workflow Design

The workflow system is split into two separate workflows to respect GitHub Actions' non-reentrancy model. GitHub Actions prevents workflows from retriggering themselves when they push commits to the same PR branch. This design uses labels to coordinate between workflows: the prepare workflow commits changes and adds a `pr_is_ready` label, which triggers the validate workflow.

**Why Two Workflows?**

1. **GitHub Actions Limitation:** When a workflow commits to a PR branch, it cannot retrigger itself on the same workflow file, even with a PAT
2. **State Transition Model:** Metadata generation is treated as a state transition - the workflow commits changes and terminates
3. **Label-Based Triggering:** The prepare workflow adds `pr_is_ready` label after successful mutations, which triggers the validate workflow via `labeled` event
4. **Idempotent Design:** The validate workflow always operates on a PR that is structurally complete (metadata exists) and only runs when `pr_is_ready` label is present

The workflow system is split into two separate workflows to respect GitHub Actions' non-reentrancy model:

1. **`plugin_submission_prepare.yml`** - Prepare phase (commits to PR, then terminates)
2. **`plugin_submission_validate.yml`** - Validate phase (handles downstream steps)

### Prepare Workflow

**`plugin_submission_prepare.yml`** - Handles all repository mutations

This workflow detects changes, validates PRs, and commits mutations to the PR branch:

1. **Detect Changes** - Identifies what plugins changed
2. **Validate PR** - Checks if PR tests pass (minimal validation to proceed)
3. **Handle Metadata-Only PR** - Adds label for metadata-only PRs, then terminates (conditional)
4. **Generate Mutations and Commit** - Single job with multiple steps:
   - Step 4a: Generate Metadata (stages files)
   - Step 4b: Layer Mapping (stages files, vision domain only)
   - Step 4c: Commit and Push (commits all staged files, pushes, terminates)

**Key Feature:** After committing and pushing, this workflow adds the `pr_is_ready` label to the PR, then terminates. The label addition triggers a `labeled` event, which starts the validate workflow.

### Validate Workflow

**`plugin_submission_validate.yml`** - Handles downstream validation and orchestration

This workflow only runs if the PR has the `pr_is_ready` label (added by prepare workflow). It assumes the PR is structurally complete (metadata exists) and handles:

1. **Check PR Ready Label** - Verifies PR has `pr_is_ready` label (gatekeeper job)
2. **Detect Changes** - Re-runs detection on normalized PR
3. **Validate PR** - Full validation for automerge eligibility
4. **Update Existing Metadata** - Triggers Jenkins for metadata-only PRs (conditional)
5. **Auto-merge** - Automatically merges approved PRs (conditional)
6. **Post-Merge Scoring** - Triggers Jenkins scoring after merge (conditional)
7. **Notify on Failure** - Sends failure notifications (always runs on failure)

### Reusable Workflows

- **`metadata_handler.yml`** - Processes plugin metadata
- **`user_notification_handler.yml`** - Handles email extraction and notifications

### Helper Scripts

- **`brainscore_language/submission/actions_helpers.py`** - Python utilities for workflow operations

## Workflow Flow

### Phase 1: Prepare Workflow (`plugin_submission_prepare.yml`)

```
PR Created/Updated
    ↓
[Prepare Workflow Starts]
    ↓
[1] Detect Changes
    ├─→ Has plugins? → Continue
    └─→ No plugins? → Skip
    ↓
[2] Validate PR (minimal validation)
    ├─→ Tests pass? → Continue
    └─→ Tests fail? → Terminate
    ↓
[3] Handle Metadata-Only PR (if metadata-only)
    ├─→ Add "only_update_metadata" and "pr_is_ready" labels
    └─→ Workflow terminates
    ↓
[4] Generate Mutations and Commit (single job with multiple steps)
    ├─→ Step 4a: Generate Metadata (if needed)
    │   └─→ Generate metadata.yml, stage files
    ├─→ Step 4b: Layer Mapping (if needed, vision only)
    │   └─→ Generate layer mapping, stage files
    └─→ Step 4c: Commit and Push
        ├─→ Commit all staged files (metadata + layer mapping)
        ├─→ Push to PR branch
        ├─→ Add "pr_is_ready" label to PR
        └─→ Workflow terminates
    ↓
[Label addition triggers labeled event]
```

### Phase 2: Validate Workflow (`plugin_submission_validate.yml`)

```
Labeled Event (pr_is_ready label added by prepare workflow)
    ↓
[Validate Workflow Starts]
    ↓
[0] Check PR Ready Label (gatekeeper)
    ├─→ Has pr_is_ready label? → Continue
    └─→ No label? → Skip all jobs
    ↓
[1] Detect Changes (re-run on normalized PR)
    ├─→ Metadata exists? → Continue
    └─→ No plugins? → Skip
    ↓
[2] Validate PR (full validation)
    ├─→ Tests pass? → Continue
    ├─→ Automergeable? → Continue
    └─→ Tests fail? → Notify user
    ↓
[3] Update Existing Metadata (if metadata-only)
    ├─→ Has only_update_metadata label? → Trigger Jenkins
    └─→ Otherwise → Skip
    ↓
[4] Auto-merge (if validated)
    ├─→ Approve PR
    └─→ Merge to main
    ↓
[5] Post-Merge Scoring (after merge)
    ├─→ Extract user email
    ├─→ Build plugin info
    └─→ Trigger Jenkins scoring
    ↓
[6] Notify on Failure (if any step fails)
    └─→ Send email notification
```

## Key Features

### Design Principles

- **Two-phase architecture** - Prepare workflow commits changes and adds label, validate workflow handles downstream steps
- **Label-based coordination** - Uses `pr_is_ready` label to coordinate between workflows instead of relying on synchronize events
- **Non-reentrant design** - Respects GitHub Actions' limitation where workflows can't retrigger themselves
- **State transition model** - Metadata generation is a terminal step that adds label to trigger validate workflow
- **Sequential execution** - Clear dependencies between steps
- **Conditional logic** - Steps only run when needed
- **Error handling** - Failure notifications for all errors
- **Reusable components** - Shared workflows for common tasks

### Structure

- **Clear job names** - Numbered steps show execution flow
- **Comprehensive outputs** - Each job exposes needed data
- **Proper permissions** - Least privilege access
- **Environment variables** - Domain-agnostic configuration
- **Documentation** - Well-commented code

### Maintainability

- **Modular structure** - Reusable workflows
- **Testable scripts** - Python helpers can be unit tested
- **Easy to extend** - Add new steps easily
- **Domain-agnostic** - Can be copied to other domains

## Usage

### For Plugin Submissions

1. Create a PR with plugin changes (models, benchmarks, metrics, data)
2. Add `automerge` or `automerge-web` label (for web submissions)
3. Workflow automatically:
   - Validates the PR
   - Runs tests
   - Merges if approved
   - Triggers scoring after merge

### For Metadata-Only Changes

1. Create a PR with only `metadata.yml` changes
2. Workflow automatically processes metadata without triggering scoring

### Manual Triggers

You can manually trigger workflows using `workflow_dispatch`:

```bash
gh workflow run metadata_handler.yml \
  --field plugin-dir="brainscore_language/models/mymodel" \
  --field plugin-type="models" \
  --field domain="language"
```

## Configuration

### Required Secrets

- `BSC_DATABASESECRET` - Database connection secret
- `AWS_ACCESS_KEY_ID` - AWS credentials
- `AWS_SECRET_ACCESS_KEY` - AWS credentials
- `JENKINS_USER` - Jenkins username
- `JENKINS_TOKEN` - Jenkins API token
- `JENKINS_TRIGGER` - Jenkins job trigger token
- `JENKINS_MAPPING_USER` - Jenkins mapping job user
- `JENKINS_MAPPING_USER_API` - Jenkins mapping job API key
- `JENKINS_MAPPING_TOKEN` - Jenkins mapping job token
- `JENKINS_MAPPING_URL` - Jenkins mapping job URL
- `GITHUB_TOKEN` - GitHub API token (auto-provided)
- `WORKFLOW_TOKEN` - Token for workflow operations
- `APPROVAL_TOKEN` - Token for PR approvals
- `GMAIL_USERNAME` - Gmail account for notifications
- `GMAIL_PASSWORD` - Gmail password for notifications
- `EMAIL_ENCRYPTION_KEY` - Key for email encryption

### Environment Variables

- `DOMAIN` - Set to "language" (default in workflow)
- `DOMAIN_ROOT` - Set to "brainscore_language" (default in workflow)
- `PYTHON_VERSION` - Set to "3.11" (default in workflow)

## Monitoring

### View Workflow Status

On any PR, you'll see two workflow runs:

**First Run (Prepare - with metadata generation):**
```
Plugin Submission Prepare
├─ 1. Detect Changes (success)
├─ 2. Validate PR (success)
├─ 3. Handle Metadata-Only PR (skipped)
└─ 4. Generate Mutations and Commit (success)
    ├─→ Step 4a: Generate Metadata (stages files)
    ├─→ Step 4b: Layer Mapping (skipped - language domain)
    └─→ Step 4c: Commit and Push (commits staged files, pushes)
        └─→ Workflow terminates, commit triggers synchronize
```

**First Run (Prepare - with layer mapping for vision):**
```
Plugin Submission Prepare
├─ 1. Detect Changes (success)
├─ 2. Validate PR (success)
├─ 3. Handle Metadata-Only PR (skipped)
└─ 4. Generate Mutations and Commit (success)
    ├─→ Step 4a: Generate Metadata (stages files)
    ├─→ Step 4b: Layer Mapping (stages files)
    └─→ Step 4c: Commit and Push (commits all staged files, pushes)
        └─→ Workflow terminates, commit triggers synchronize
```

**Second Run (Validate):**
```
Plugin Submission Validate
├─ 1. Detect Changes (success)
├─ 2. Validate PR (success)
├─ 3. Auto-merge (success)
└─ 4. Post-Merge Scoring (success)
```

### Debugging

1. Check workflow logs in GitHub Actions
2. Review job outputs for each step
3. Check Jenkins for scoring job status
4. Review email notifications for failures

## Comparison to Vision

This workflow system learns from vision's mistakes:

| Aspect | Vision (Old) | Language (New) |
|--------|-------------|----------------|
| **Workflows** | 10 separate files | 2 main workflows + 2 reusable |
| **Architecture** | Single workflow | Two-phase (prepare + validate) |
| **Entry Point** | Multiple triggers | Clear separation of concerns |
| **Dependencies** | Implicit, complex | Explicit, sequential |
| **Visibility** | Hard to see flow | Clear numbered steps |
| **Error Handling** | Scattered | Centralized |
| **Re-entrancy** | Attempts impossible continuation | Respects GitHub Actions model |
| **Maintainability** | Hard to modify | Easy to extend |

## Future Improvements

Potential enhancements:

1. **Add Airflow** - Migrate to Airflow for even better orchestration
2. **Metrics Dashboard** - Track workflow performance
3. **Retry Logic** - Automatic retries for transient failures
4. **Caching** - Cache plugin detection results
5. **Parallel Execution** - Run independent steps in parallel

## Support

For issues or questions:
- Check workflow logs in GitHub Actions
- Review this README
- Consult the main Brain-Score documentation
- Contact the Brain-Score team

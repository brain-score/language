# Language Domain Workflow Orchestration

## Overview

This directory contains a unified workflow for LLM evaluations in the Brain-Score language domain. The system uses a single orchestrated workflow that handles all steps from change detection through post-merge scoring.

## Architecture

### Unified Workflow Design

The workflow system uses a single unified workflow that runs all steps sequentially in one workflow run:

1. **`plugin_submission_orchestrator.yml`** - Unified workflow handling all phases

### Orchestrator Workflow

**`plugin_submission_orchestrator.yml`** - Handles all repository operations

This workflow detects changes, validates PRs, commits mutations, and handles downstream steps:

1. **Detect Changes** - Identifies what plugins changed
2. **Validate PR** - Checks if PR tests pass (minimal validation to proceed)
3. **Handle Metadata-Only PR** - Adds label for metadata-only PRs (conditional)
4. **Generate Mutations and Commit** - Single job with multiple steps:
   - Step 4a: Generate Metadata (stages files)
   - Step 4b: Layer Mapping (stages files, vision domain only)
   - Step 4c: Commit and Push (commits all staged files, pushes)
5. **Update Existing Metadata** - Triggers Jenkins for metadata-only PRs (conditional)
6. **Auto-merge** - Automatically merges approved PRs (conditional)
7. **Post-Merge Scoring** - Triggers Jenkins scoring after merge (conditional)
8. **Notify on Failure** - Sends failure notifications (always runs on failure)

### Reusable Workflows

- **`metadata_handler.yml`** - Processes plugin metadata
- **`user_notification_handler.yml`** - Handles email extraction and notifications

### Helper Scripts

- **`brainscore_language/submission/actions_helpers.py`** - Python utilities for workflow operations

## Workflow Flow

### Unified Orchestrator Workflow (`plugin_submission_orchestrator.yml`)

```
PR Created/Updated
    ↓
[Orchestrator Workflow Starts]
    ↓
[1] Detect Changes
    ├─→ Has plugins? → Continue
    └─→ No plugins? → Skip downstream jobs
    ↓
[2] Validate PR (minimal validation)
    ├─→ Tests pass? → Continue
    └─→ Tests fail? → Skip mutation jobs
    ↓
[3] Handle Metadata-Only PR (if metadata-only)
    ├─→ Add "only_update_metadata" and "pr_is_ready" labels
    └─→ Continue to downstream jobs
    ↓
[4] Generate Mutations and Commit (if needed)
    ├─→ Step 4a: Generate Metadata (if needed)
    │   └─→ Generate metadata.yml, stage files
    ├─→ Step 4b: Layer Mapping (if needed, vision only)
    │   └─→ Generate layer mapping, stage files
    └─→ Step 4c: Commit and Push
        ├─→ Commit all staged files (metadata + layer mapping)
        ├─→ Push to PR branch
        └─→ Add "pr_is_ready" label to PR
    ↓
[5] Update Existing Metadata (if metadata-only)
    ├─→ Has only_update_metadata label? → Trigger Jenkins
    └─→ Otherwise → Skip
    ↓
[6] Auto-merge (if validated)
    ├─→ Approve PR
    └─→ Merge to main
    ↓
[7] Post-Merge Scoring (after merge)
    ├─→ Extract user email
    ├─→ Build plugin info
    └─→ Trigger Jenkins scoring
    ↓
[8] Notify on Failure (if any step fails)
    └─→ Send email notification
```

## Key Features

### Design Principles

- **Unified architecture** - Single workflow handles all phases from detection to post-merge scoring
- **Sequential execution** - Clear dependencies between steps, all jobs run in one workflow run
- **Conditional logic** - Steps only run when needed based on PR state
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

On any PR, you'll see a single unified workflow run:

**Orchestrator Workflow (with metadata generation):**
```
Plugin Submission Orchestrator
├─ 1. Detect Changes (success)
├─ 2. Validate PR (success)
├─ 3. Handle Metadata-Only PR (skipped)
├─ 4. Generate Mutations and Commit (success)
│   ├─→ Step 4a: Generate Metadata (stages files)
│   ├─→ Step 4b: Layer Mapping (skipped - language domain)
│   └─→ Step 4c: Commit and Push (commits staged files, pushes)
├─ 5. Update Existing Metadata (skipped)
├─ 6. Auto-merge (success)
├─ 7. Post-Merge Scoring (success)
└─ 8. Notify on Failure (skipped - no failures)
```

**Orchestrator Workflow (with layer mapping for vision):**
```
Plugin Submission Orchestrator
├─ 1. Detect Changes (success)
├─ 2. Validate PR (success)
├─ 3. Handle Metadata-Only PR (skipped)
├─ 4. Generate Mutations and Commit (success)
│   ├─→ Step 4a: Generate Metadata (stages files)
│   ├─→ Step 4b: Layer Mapping (stages files)
│   └─→ Step 4c: Commit and Push (commits all staged files, pushes)
├─ 5. Update Existing Metadata (skipped)
├─ 6. Auto-merge (success)
├─ 7. Post-Merge Scoring (success)
└─ 8. Notify on Failure (skipped - no failures)
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
| **Workflows** | 10 separate files | 1 unified workflow + 2 reusable |
| **Architecture** | Single workflow | Unified orchestrator |
| **Entry Point** | Multiple triggers | Single entry point |
| **Dependencies** | Implicit, complex | Explicit, sequential |
| **Visibility** | Hard to see flow | Clear numbered steps |
| **Error Handling** | Scattered | Centralized |
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

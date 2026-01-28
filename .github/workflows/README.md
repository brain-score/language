# Language Domain Workflow Orchestration

## Overview

This directory contains two complementary workflows for LLM evaluations in the Brain-Score language domain. The system uses a two-phase approach: mutation (commits changes to PR) and orchestration (handles downstream steps after PR is complete).

## Architecture

### Two-Phase Workflow Design

The workflow system is split into two separate workflows to respect GitHub Actions' non-reentrancy model. GitHub Actions prevents workflows from retriggering themselves when they push commits to the same PR branch. This design ensures mutations (metadata generation, layer mapping) are committed and the workflow terminates, then a fresh orchestrator run handles downstream steps.

**Why Two Workflows?**

1. **GitHub Actions Limitation:** When a workflow commits to a PR branch, it cannot retrigger itself on the same workflow file, even with a PAT
2. **State Transition Model:** Metadata generation is treated as a state transition - the workflow commits changes and terminates
3. **Fresh Orchestrator Run:** The commit triggers a `synchronize` event, which starts a fresh orchestrator workflow run
4. **Idempotent Design:** The orchestrator always operates on a PR that is structurally complete (metadata exists)

The workflow system is split into two separate workflows to respect GitHub Actions' non-reentrancy model:

1. **`plugin_submission_mutate.yml`** - Mutation phase (commits to PR, then terminates)
2. **`plugin_submission_orchestrator.yml`** - Orchestration phase (handles downstream steps)

### Mutation Workflow

**`plugin_submission_mutate.yml`** - Handles all repository mutations

This workflow detects changes, validates PRs, and commits mutations to the PR branch:

1. **Detect Changes** - Identifies what plugins changed
2. **Validate PR** - Checks if PR tests pass (minimal validation to proceed)
3. **Handle Metadata-Only PR** - Adds label for metadata-only PRs, then terminates (conditional)
4. **Generate Mutations and Commit** - Single job with multiple steps:
   - Step 4a: Generate Metadata (stages files)
   - Step 4b: Layer Mapping (stages files, vision domain only)
   - Step 4c: Commit and Push (commits all staged files, pushes, terminates)

**Key Feature:** After committing and pushing, this workflow terminates. The commit triggers a `synchronize` event, which starts the orchestrator workflow.

### Orchestration Workflow

**`plugin_submission_orchestrator.yml`** - Handles downstream orchestration

This workflow assumes the PR is structurally complete (metadata exists) and handles:

1. **Detect Changes** - Re-runs detection on normalized PR
2. **Validate PR** - Full validation for automerge eligibility
3. **Auto-merge** - Automatically merges approved PRs (conditional)
4. **Post-Merge Scoring** - Triggers Jenkins scoring after merge (conditional)
5. **Notify on Failure** - Sends failure notifications (always runs on failure)

### Reusable Workflows

- **`metadata_handler.yml`** - Processes plugin metadata
- **`user_notification_handler.yml`** - Handles email extraction and notifications

### Helper Scripts

- **`brainscore_language/submission/actions_helpers.py`** - Python utilities for workflow operations

## Workflow Flow

### Phase 1: Mutation Workflow (`plugin_submission_mutate.yml`)

```
PR Created/Updated
    ↓
[Mutation Workflow Starts]
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
    ├─→ Add "only_update_metadata" label
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
        └─→ Workflow terminates
    ↓
[Commit triggers synchronize event]
```

### Phase 2: Orchestration Workflow (`plugin_submission_orchestrator.yml`)

```
Synchronize Event (from mutation commit)
    ↓
[Orchestration Workflow Starts]
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
[3] Auto-merge (if validated)
    ├─→ Approve PR
    └─→ Merge to main
    ↓
[4] Post-Merge Scoring (after merge)
    ├─→ Extract user email
    ├─→ Build plugin info
    └─→ Trigger Jenkins scoring
    ↓
[5] Notify on Failure (if any step fails)
    └─→ Send email notification
```

## Key Features

### Design Principles

- **Two-phase architecture** - Mutation workflow commits changes, orchestrator handles downstream steps
- **Non-reentrant design** - Respects GitHub Actions' limitation where workflows can't retrigger themselves
- **State transition model** - Metadata generation is a terminal step that triggers fresh orchestrator run
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

**First Run (Mutation - with metadata generation):**
```
Plugin Submission Mutate
├─ 1. Detect Changes (success)
├─ 2. Validate PR (success)
├─ 3. Handle Metadata-Only PR (skipped)
└─ 4. Generate Mutations and Commit (success)
    ├─→ Step 4a: Generate Metadata (stages files)
    ├─→ Step 4b: Layer Mapping (skipped - language domain)
    └─→ Step 4c: Commit and Push (commits staged files, pushes)
        └─→ Workflow terminates, commit triggers synchronize
```

**First Run (Mutation - with layer mapping for vision):**
```
Plugin Submission Mutate
├─ 1. Detect Changes (success)
├─ 2. Validate PR (success)
├─ 3. Handle Metadata-Only PR (skipped)
└─ 4. Generate Mutations and Commit (success)
    ├─→ Step 4a: Generate Metadata (stages files)
    ├─→ Step 4b: Layer Mapping (stages files)
    └─→ Step 4c: Commit and Push (commits all staged files, pushes)
        └─→ Workflow terminates, commit triggers synchronize
```

**Second Run (Orchestration):**
```
Plugin Submission Orchestrator
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
| **Architecture** | Single workflow | Two-phase (mutation + orchestration) |
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

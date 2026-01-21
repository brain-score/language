# Language Domain Workflow Orchestration

## Overview

This directory contains a **professional, industry-standard orchestration workflow** for LLM evaluations in the Brain-Score language domain. The system consolidates all plugin submission logic into a single, clear, sequential flow.

## Architecture

### Main Orchestrator

**`plugin_submission_orchestrator.yml`** - Single entry point for all plugin submissions

This workflow orchestrates the entire submission pipeline:

1. **Detect Changes** - Identifies what plugins changed
2. **Validate PR** - Checks if PR is automergeable (pre-merge only)
3. **Layer Mapping** - Maps layers for new models (conditional)
4. **Process Metadata** - Handles metadata-only changes (conditional)
5. **Auto-merge** - Automatically merges approved PRs (conditional)
6. **Post-Merge Scoring** - Triggers Jenkins scoring after merge (conditional)
7. **Notify on Failure** - Sends failure notifications (always runs on failure)

### Reusable Workflows

- **`metadata_handler.yml`** - Processes plugin metadata
- **`user_notification_handler.yml`** - Handles email extraction and notifications

### Helper Scripts

- **`brainscore_language/submission/actions_helpers.py`** - Python utilities for workflow operations

## Workflow Flow

```
PR Created/Updated
    ↓
[1] Detect Changes
    ├─→ Has plugins? → Continue
    └─→ No plugins? → Skip
    ↓
[2] Validate PR (pre-merge only)
    ├─→ Tests pass? → Continue
    └─→ Tests fail? → Notify user
    ↓
[3] Layer Mapping (if new models)
    ├─→ Trigger Jenkins mapping
    └─→ Wait for completion
    ↓
[4] Process Metadata (if metadata-only)
    ├─→ Process metadata
    └─→ Create/merge metadata PRs
    ↓
[5] Auto-merge (if validated)
    ├─→ Approve PR
    └─→ Merge to main
    ↓
[6] Post-Merge Scoring (after merge)
    ├─→ Extract user email
    ├─→ Build plugin info
    └─→ Trigger Jenkins scoring
    ↓
[7] Notify on Failure (if any step fails)
    └─→ Send email notification
```

## Key Features

### ✅ Industry Standard

- **Single orchestrator** - One workflow to rule them all
- **Sequential execution** - Clear dependencies between steps
- **Conditional logic** - Steps only run when needed
- **Error handling** - Comprehensive failure notifications
- **Reusable components** - DRY principle applied

### ✅ Professional Design

- **Clear job names** - Numbered steps show flow
- **Comprehensive outputs** - Each job exposes needed data
- **Proper permissions** - Least privilege access
- **Environment variables** - Domain-agnostic configuration
- **Documentation** - Well-commented code

### ✅ Maintainable

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

On any PR, you'll see a single workflow run:

```
Plugin Submission Orchestrator
├─ 1. Detect Changes ✅
├─ 2. Validate PR ✅
├─ 3. Layer Mapping ⏭️ (skipped)
├─ 4. Process Metadata ⏭️ (skipped)
├─ 5. Auto-merge ✅
└─ 6. Post-Merge Scoring ✅
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
| **Workflows** | 10 separate files | 1 orchestrator + 2 reusable |
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

# Brain-Score Language: Scoring Flow Documentation

## Overview

This document explains the complete scoring flow for the Brain-Score language domain. The system uses a unified workflow architecture to automatically score language models on benchmarks when plugins (models or benchmarks) are submitted via pull requests.

### Unified Architecture

The workflow system uses a single unified workflow:

1. **Orchestrator Workflow** (`plugin_submission_orchestrator.yml`): Handles all phases from change detection through post-merge scoring in a single workflow run

This unified design runs all jobs sequentially in one workflow execution, eliminating the need for label-based coordination between separate workflows.

## High-Level Flow

```
PR Created/Updated
    ↓
[Orchestrator Workflow]
    ├─→ Detect Changes
    ├─→ Validate PR (minimal validation)
    ├─→ Handle Metadata-Only PR (if metadata-only changes)
    │   └─→ Add "only_update_metadata" and "pr_is_ready" labels
    ├─→ Generate Mutations and Commit (if new plugins need metadata or mapping)
    │   ├─→ Step 4a: Generate Metadata (stages files)
    │   ├─→ Step 4b: Layer Mapping (stages files, vision only)
    │   └─→ Step 4c: Commit and Push
    │       ├─→ Commit all staged files
    │       ├─→ Push to PR branch
    │       └─→ Add "pr_is_ready" label to PR
    ├─→ Update Existing Metadata (if metadata-only)
    ├─→ Auto-merge (if approved)
    └─→ Post-Merge Scoring (after merge)
        └─→ Jenkins Scoring Job
            └─→ Results Stored in Database
```

## Detailed Workflow Steps

### 1. PR Creation/Update

**Trigger:** When a pull request is:
- Created (`opened`)
- Updated (`synchronize`)
- Labeled (`labeled` with `automerge` or `automerge-web`)

**What Happens:**
- The `plugin_submission_orchestrator.yml` workflow is triggered
- Workflow detects what files changed in the PR
- If mutations are needed (metadata generation, layer mapping), they are committed
- All downstream steps (validation, automerge, scoring) run in the same workflow

### 2. Detect Changes

**Job:** `1. Detect Changes`

**Purpose:** Identify what plugins (models, benchmarks, metrics, data) were modified or added

**Process:**
1. Compares changed files between PR head and base branch
2. Uses `brainscore_core.plugin_management.parse_plugin_changes.get_scoring_info()` to analyze changes
3. Determines:
   - Whether plugins were modified
   - Plugin type (models, benchmarks, etc.)
   - Whether new models were added (needs layer mapping)
   - Whether only metadata changed (skip scoring)
   - Whether scoring is needed

**Outputs:**
- `has_plugins`: Boolean - whether any plugins were changed
- `plugin_type`: String - "models" or "benchmarks"
- `plugin_dirs`: String - comma-separated plugin directory paths
- `has_new_models`: Boolean - whether new models were added
- `metadata_only`: Boolean - whether only metadata.yml changed
- `needs_scoring`: Boolean - whether scoring should be triggered
- `needs_mapping`: Boolean - whether layer mapping is needed
- `needs_metadata_generation`: Boolean - whether plugins need metadata.yml generated
- `is_automergeable`: Boolean - whether PR only changes plugins

### 3. Validate PR (Pre-Merge)

**Job:** `2. Validate PR`

**When:** Only runs for `pull_request` events (not post-merge)

**Purpose:** Check if PR is eligible for auto-merge

**Process:**
1. Checks if PR has `automerge` or `automerge-web` label
2. Verifies all required tests pass:
   - "Language Unittests, Plugins"
   - "Language Unittests, Non-Plugins"
   - "Language Integration Tests"
3. Confirms PR only modifies plugin files (no core code changes)

**Outputs:**
- `is_automergeable`: Boolean - whether PR can be auto-merged
- `all_tests_pass`: Boolean - whether all tests passed
- `pr_number`: Integer - PR number
- `is_automerge_web`: Boolean - whether this is a web submission

**Next Steps:**
- If valid → proceed to layer mapping (if needed) or auto-merge
- If invalid → notify user of failure

### 4. Handle Metadata-Only PR (Conditional)

**Job:** `3. Handle Metadata-Only PR`

**When:** Only runs if `metadata_only == true` (only metadata.yml changed, no code changes)

**Purpose:** Add label to PR for downstream metadata update job

**Process:**
1. Detects that PR only contains metadata.yml changes
2. Adds "only_update_metadata" label to PR
3. Adds "pr_is_ready" label to PR (for informational purposes)
4. Workflow continues to downstream jobs

**Note:**
- This job adds labels but doesn't terminate the workflow
- The "Update Existing Metadata" job will see the `only_update_metadata` label and trigger Jenkins job
- Different from "Generate Mutations and Commit" which creates new metadata files

### 5. Generate Mutations and Commit (Conditional)

**Job:** `4. Generate Mutations and Commit`

**When:** Only runs if:
- New plugins need metadata generation (`needs_metadata_generation == true`), OR
- New models need layer mapping (`needs_mapping == true`)
- All tests passed (`all_tests_pass == true`)

**Purpose:** Generate metadata and/or layer mapping files, then commit and push all mutations together

**Process:**

**Step 4a: Generate Metadata** (if `needs_metadata_generation == true`)
1. Checks each plugin directory for existing metadata files
2. For plugins missing metadata:
   - Generates `metadata.yml` using domain-specific metadata generators
   - Extracts model/benchmark information automatically
   - Creates metadata file in plugin directory
   - **Stages the file** with `git add` (does not commit)

**Step 4b: Layer Mapping** (if `needs_mapping == true`, vision domain only)
1. Triggers Jenkins layer mapping job
2. Layer mapping files are generated
   - **Stages the files** with `git add` (does not commit)

**Step 4c: Commit and Push**
1. Commits all staged files from steps 4a and 4b in a single commit
2. Pushes commit to PR branch using PAT (or GITHUB_TOKEN if PAT not configured)
3. Adds `pr_is_ready` label to PR (for informational purposes)
4. Workflow continues to downstream jobs

**Note:** 
- All steps run in the same job, so staged files persist across steps
- No intermediate commits - everything is committed together in step 4c
- The `pr_is_ready` label is added for informational purposes
- Workflow continues to downstream validation and automerge jobs

**Metadata Generation:**
- For models: Uses `ModelMetadataGenerator` to extract:
  - Architecture information
  - Parameter counts
  - Model family
  - Other model-specific metadata
- For benchmarks: Uses `BenchmarkMetadataGenerator` to extract:
  - Stimulus set information
  - Data metadata
  - Metric information

**Layer Mapping (vision only):**
- Triggers Jenkins job to map model layers
- Maps model architecture layers
- Updates model metadata

### 6. Update Existing Metadata (Conditional)

**Job:** `5. Update Existing Metadata`

**When:** Only runs if:
- PR has "only_update_metadata" label (added by Handle Metadata-Only PR job)
- `metadata_only == true`

**Purpose:** Trigger Jenkins job to update existing metadata in database

**Process:**
1. Checks for "only_update_metadata" label on PR
2. Triggers Jenkins job "update_existing_metadata" via `actions_helpers.py trigger_update_existing_metadata`
3. Passes plugin information to Jenkins:
   - Plugin directories
   - Plugin type
   - Domain

**Jenkins Job:**
- Updates existing metadata in database
- Processes metadata.yml files
- No commits made to PR

**Note:**
- This step only runs if Handle Metadata-Only PR job detected metadata-only PR and added the label
- Different from "Generate Mutations and Commit" which creates new metadata files

### 7. Auto-merge (Conditional)

**Job:** `6. Auto-merge`

**When:** Only runs if:
- PR is validated (`is_automergeable == true`)
- All tests pass (`all_tests_pass == true`)
- PR is structurally complete (metadata exists)
- Update Existing Metadata completed (or was skipped)

**Purpose:** Automatically merge approved PRs

**Process:**
1. Auto-approves the PR using `hmarr/auto-approve-action`
2. Merges PR using `plm9606/automerge_actions`:
   - GitHub submissions: merges PRs with `automerge` label
   - Web submissions: merges PRs with `automerge-web` label
   - Uses squash merge method
   - Deletes branch after merge

**Result:** PR is merged to `main` branch

### 8. Post-Merge Scoring

**Job:** `7. Post-Merge Scoring`

**When:** Only runs if:
- PR was merged to `main` (`pull_request_target` event, `merged == true`)
- Scoring is needed (`needs_scoring == True`)
- Not metadata-only (`metadata_only == false`)

**Purpose:** Trigger Jenkins to score models on benchmarks

**Process:**

1. **Extract User Email:**
   - For web submissions: extracts user ID from PR title, looks up email in database
   - For GitHub submissions: gets email from GitHub user profile
   - Masks email in logs for security

2. **Build Plugin Info:**
   - Combines plugin information into JSON payload:
     ```json
     {
       "domain": "language",
       "email": "user@example.com",
       "plugin_dirs": "brainscore_language/models/mymodel",
       "plugin_type": "models",
       "competition": "None",
       "model_type": "artificialsubject",
       "public": true,
       "new_models": [...],
       "new_benchmarks": [...],
       ...
     }
     ```

3. **Trigger Jenkins:**
   - Calls `brainscore_core.submission.endpoints.call_jenkins()`
   - Sends plugin info to Jenkins job `dev_score_plugins`
   - Jenkins job URL: `http://www.brain-score-jenkins.com:8080/job/dev_score_plugins/buildWithParameters`

**Jenkins Scoring Job:**

The Jenkins job receives the plugin information and:

1. **Resolves Models and Benchmarks:**
   - If `new_models` specified: scores those models on all public benchmarks
   - If `new_benchmarks` specified: scores all public models on those benchmarks
   - If both specified: scores all public models on all public benchmarks
   - Uses `resolve_models_benchmarks()` function

2. **For Each Model-Benchmark Pair:**
   - Loads model using `brainscore_language.load_model()`
   - Loads benchmark using `brainscore_language.load_benchmark()`
   - Runs scoring using `brainscore_language.score()`
   - Stores results in database

3. **Database Storage:**
   - Creates/updates `Model` entry
   - Creates/updates `BenchmarkInstance` entry
   - Creates/updates `Score` entry with:
     - Score value
     - Start/end timestamps
     - Error messages (if failed)

### 9. Failure Notification

**Job:** `8. Notify on Failure`

**When:** Runs if any job in the workflow fails

**Purpose:** Notify users when their submission fails

**Process:**
1. Determines which job failed
2. Extracts user email (same process as post-merge scoring)
3. Sends failure notification email via `actions_helpers.py send_failure_email`
4. Adds `failure-notified` label to PR to prevent duplicate notifications

**Email Content:**
- Subject: "Brain-Score Language Submission Failed"
- Body includes:
  - Failure reason
  - Link to PR
  - Instructions to fix or resubmit

## Workflow File Structure

```
.github/workflows/
├── plugin_submission_orchestrator.yml   # Unified orchestrator workflow
├── metadata_handler.yml                  # Reusable: processes metadata
└── user_notification_handler.yml        # Reusable: handles notifications

brainscore_language/submission/
└── actions_helpers.py                    # Python utilities for workflows
```

## Key Components

### Orchestrator Workflow

**File:** `.github/workflows/plugin_submission_orchestrator.yml`

**Key Features:**
- Handles all repository operations in a single unified workflow
- Detects changes, validates PRs, commits mutations, and handles downstream steps
- Sequential job execution with clear dependencies
- Conditional logic for different scenarios
- Comprehensive error handling
- All jobs run in one workflow execution

### Helper Script

**File:** `brainscore_language/submission/actions_helpers.py`

**Functions:**
- `validate_pr()` - Validates PR for automerge eligibility
- `trigger_update_existing_metadata()` - Triggers Jenkins update_existing_metadata job
- `trigger_layer_mapping()` - Triggers Jenkins layer mapping job
- `extract_email()` - Extracts user email from PR or database
- `send_failure_email()` - Sends failure notification emails

### Jenkins Integration

**Jenkins Job:** `dev_score_plugins`

**Trigger:** Called via `call_jenkins()` function with plugin info JSON

**Process:**
1. Receives plugin information as parameters
2. Resolves which models/benchmarks to score
3. Runs scoring for each model-benchmark pair
4. Stores results in database
5. Reports status back to GitHub

## Example Scenarios

### Scenario 1: New Model Submission (Without Metadata)

**Orchestrator Workflow:**
1. User creates PR with new model in `brainscore_language/models/newmodel/`
   - Model code is added but no `metadata.yml` file
2. Workflow detects new model
3. Validates PR (tests must pass)
4. **Generate Mutations and Commit job** runs:
   - **Step 4a:** Generates metadata.yml for the new model (stages file)
   - **Step 4b:** Skips layer mapping (language domain doesn't require it)
   - **Step 4c:** Commits staged metadata file, pushes to PR branch, adds `pr_is_ready` label
5. Workflow continues to downstream jobs
6. Validates PR (full validation)
7. Auto-merges PR
8. Post-merge: triggers scoring
   - New model scored on all public benchmarks
   - Results stored in database

### Scenario 1b: New Model Submission (With Metadata)

**Orchestrator Workflow:**
1. User creates PR with new model in `brainscore_language/models/newmodel/`
   - Model code AND `metadata.yml` file are both provided
2. Workflow detects new model
3. Validates PR (tests must pass)
4. **Generate Mutations and Commit job** is skipped (no mutations needed)
5. Workflow continues to downstream jobs
6. Validates PR (full validation)
7. Auto-merges PR
8. Post-merge: triggers scoring

### Scenario 2: New Benchmark Submission

**Orchestrator Workflow:**
1. User creates PR with new benchmark in `brainscore_language/benchmarks/newbenchmark/`
   - Benchmark code added (with or without metadata.yml)
2. Workflow detects new benchmark
3. Validates PR (tests must pass)
4. **Generate Mutations and Commit job** runs:
   - **Step 4a:** Generates metadata.yml if missing (stages file)
   - **Step 4b:** Skips layer mapping (not needed for benchmarks)
   - **Step 4c:** Commits staged metadata file, pushes to PR branch, adds `pr_is_ready` label
5. Workflow continues to downstream jobs
6. Validates PR (full validation)
7. Auto-merges PR
8. Post-merge: triggers scoring
   - All public models scored on new benchmark
   - Results stored in database

### Scenario 3: Metadata-Only Update

**Orchestrator Workflow:**
1. User creates PR with only `metadata.yml` changes
2. Workflow detects metadata-only change (`metadata_only == true`)
3. Validates PR (tests must pass)
4. **Handle Metadata-Only PR job** runs:
   - Adds "only_update_metadata" label to PR
   - Adds "pr_is_ready" label to PR
5. Workflow continues to downstream jobs
6. **Update Existing Metadata job** runs:
   - Detects "only_update_metadata" label
   - Triggers Jenkins job "update_existing_metadata"
   - Jenkins updates database with new metadata
7. Auto-merges PR (if validated)
8. Post-merge scoring skipped (`metadata_only == true`)

### Scenario 4: Web Submission

1. User submits via brain-score.org website
2. Website creates PR with `automerge-web` label
3. PR title includes user ID: `(user:12345)`
4. Workflow extracts email from database using user ID
5. Same flow as GitHub submission, but uses database for email lookup

## Database Schema

Scoring results are stored in the Brain-Score database:

- **Model** - Model entries
- **ModelMeta** - Model metadata
- **BenchmarkInstance** - Benchmark instances
- **Score** - Scoring results linking models to benchmarks
- **Submission** - Submission tracking

## Status Checks

The workflow integrates with GitHub status checks:

- **Language Unittests, Plugins** - Tests for plugin code
- **Language Unittests, Non-Plugins** - Tests for core code
- **Language Integration Tests** - Integration tests for language domain

These must pass before auto-merge is allowed.

## Labels

The workflow uses labels to control behavior:

- **`automerge`** - Standard GitHub PR auto-merge
- **`automerge-web`** - Web submission auto-merge
- **`pr_is_ready`** - Added after successful mutations (informational)
- **`only_update_metadata`** - Added for metadata-only PRs (triggers Jenkins update job)
- **`failure-notified`** - User has been notified of failure (prevents duplicate emails)

## Secrets Required

The workflow requires these GitHub repository secrets:

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

## Monitoring

### View Workflow Status

On any PR, you'll see a single unified workflow run:

**Orchestrator Workflow (if mutations needed):**
```
Plugin Submission Orchestrator
├─ 1. Detect Changes (success)
├─ 2. Validate PR (success)
├─ 3. Handle Metadata-Only PR (skipped)
├─ 4. Generate Mutations and Commit (success)
│   ├─→ Step 4a: Generate Metadata (stages files)
│   ├─→ Step 4b: Layer Mapping (skipped - language domain)
│   └─→ Step 4c: Commit and Push
│       ├─→ Commits staged files, pushes
│       └─→ Adds "pr_is_ready" label to PR
├─ 5. Update Existing Metadata (skipped)
├─ 6. Auto-merge (success)
├─ 7. Post-Merge Scoring (success)
└─ 8. Notify on Failure (skipped - no failures)
```

**Orchestrator Workflow (if no mutations needed):**
```
Plugin Submission Orchestrator
├─ 1. Detect Changes (success)
├─ 2. Validate PR (success)
├─ 3. Handle Metadata-Only PR (skipped)
├─ 4. Generate Mutations and Commit (skipped - no mutations needed)
├─ 5. Update Existing Metadata (skipped)
├─ 6. Auto-merge (success)
├─ 7. Post-Merge Scoring (success)
└─ 8. Notify on Failure (skipped - no failures)
```

### Check Jenkins Status

Jenkins job status can be viewed at:
- Jenkins URL: `http://www.brain-score-jenkins.com:8080`
- Job: `dev_score_plugins`

### View Scoring Results

Scoring results are stored in the Brain-Score database and can be viewed:
- Via the Brain-Score website
- Directly in the database
- Through the Brain-Score API

## Troubleshooting

### PR Not Auto-Merging

**Check:**
1. Does PR have `automerge` or `automerge-web` label?
2. Are all tests passing?
3. Does PR only modify plugin files?
4. Check workflow logs for validation errors

### Scoring Not Triggered

**Check:**
1. Was PR merged to `main`?
2. Did PR modify models or benchmarks?
3. Was it metadata-only? (scoring skipped)
4. Check Jenkins job status
5. Review workflow logs for errors

### Layer Mapping Not Running

**Check:**
1. **For language domain:** This is expected - layer mapping is automatically skipped
2. Are new models actually added? (not just modified)
3. Did tests pass? (mapping requires passing tests)
4. Check Jenkins mapping job status (vision domain only)
5. Review workflow logs - should show "Layer mapping skipped: language domain does not require layer mapping"

### Metadata Not Generated

**Check:**
1. Does the plugin directory actually exist?
2. Does metadata.yml already exist? (generation skipped if present)
3. Check if language domain plugin is properly configured
4. Review metadata generation logs for errors
5. Verify plugin is properly registered in the domain

### Email Notifications Not Sent

**Check:**
1. Is email properly extracted? (check logs)
2. Are Gmail credentials correct?
3. Check for `failure-notified` label (prevents duplicates)
4. Review email sending logs

## Best Practices

1. **Always test locally** before submitting PRs
2. **Ensure tests pass** before expecting auto-merge
3. **Use descriptive PR titles** (especially for web submissions with user ID)
4. **Check workflow status** on PR page
5. **Monitor Jenkins jobs** for scoring progress
6. **Review database** for scoring results

## Related Documentation

- [Workflow README](.github/workflows/README.md) - Detailed workflow documentation
- [Submission System](../core/brainscore_core/submission/developers_guide.md) - Core submission system docs
- [Plugin Management](../core/brainscore_core/plugin_management/README.md) - Plugin system docs

## Support

For issues or questions:
- Check workflow logs in GitHub Actions
- Review Jenkins job logs
- Consult Brain-Score team
- Check Brain-Score documentation

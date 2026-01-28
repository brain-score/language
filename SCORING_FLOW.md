# Brain-Score Language: Scoring Flow Documentation

## Overview

This document explains the complete scoring flow for the Brain-Score language domain. The system uses a two-phase workflow architecture to automatically score language models on benchmarks when plugins (models or benchmarks) are submitted via pull requests.

### Two-Phase Architecture

The workflow system is split into two complementary workflows:

1. **Mutation Workflow** (`plugin_submission_mutate.yml`): Handles repository mutations (metadata generation, layer mapping) and commits changes to the PR branch, then terminates
2. **Orchestration Workflow** (`plugin_submission_orchestrator.yml`): Handles downstream orchestration (automerge, post-merge scoring, notifications) and assumes the PR is structurally complete

This design respects GitHub Actions' non-reentrancy model, where workflows cannot retrigger themselves when they push commits to the same PR branch. The mutation workflow commits changes and terminates, then the commit triggers a `synchronize` event that starts a fresh orchestrator workflow run.

## High-Level Flow

```
PR Created/Updated
    ↓
[Phase 1: Mutation Workflow]
    ├─→ Detect Changes
    ├─→ Validate PR (minimal validation)
    ├─→ Update Existing Metadata (if metadata-only changes)
    ├─→ Generate Metadata (if new plugins without metadata.yml)
    ├─→ Layer Mapping (if new models, vision only - skipped for language)
    └─→ Commit and Push → Workflow terminates
        ↓
    [Commit triggers synchronize event]
        ↓
[Phase 2: Orchestration Workflow]
    ├─→ Detect Changes (re-run on normalized PR)
    ├─→ Validate PR (full validation)
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
- The `plugin_submission_mutate.yml` workflow is triggered first
- Workflow detects what files changed in the PR
- If mutations are needed (metadata generation, layer mapping), they are committed
- The commit triggers a `synchronize` event
- The `plugin_submission_orchestrator.yml` workflow then runs on the normalized PR

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
3. Confirms PR only modifies plugin files (no core code changes)

**Outputs:**
- `is_automergeable`: Boolean - whether PR can be auto-merged
- `all_tests_pass`: Boolean - whether all tests passed
- `pr_number`: Integer - PR number
- `is_automerge_web`: Boolean - whether this is a web submission

**Next Steps:**
- If valid → proceed to layer mapping (if needed) or auto-merge
- If invalid → notify user of failure

### 4. Update Existing Metadata (Conditional)

**Job:** `3. Update Existing Metadata`

**When:** Only runs if `metadata_only == true` (only metadata.yml changed, no code changes)

**Purpose:** Process metadata-only updates without triggering full scoring

**Process:**
1. Calls reusable `metadata_handler.yml` workflow
2. Processes metadata.yml files for affected plugins
3. Validates metadata structure
4. Updates database with new metadata
5. Creates PRs for metadata updates if needed
6. Auto-approves and merges metadata PRs

**Note:** 
- This step is skipped if code changes were made (not metadata-only)
- Different from "Generate Metadata" which creates new metadata files
- This is for when users only want to update existing metadata

### 5. Generate Metadata (Conditional)

**Job:** `4. Generate Metadata` (in mutation workflow)

**When:** Only runs if:
- New plugins were added (`has_plugins == true`)
- Not metadata-only (`metadata_only == false`)
- Plugin directories are missing `metadata.yml` or `metadata.yaml`
- All tests passed (`all_tests_pass == true`)

**Purpose:** Automatically generate metadata.yml files for new plugins that don't have one

**Process:**
1. Checks each plugin directory for existing metadata files
2. For plugins missing metadata:
   - Generates `metadata.yml` using domain-specific metadata generators
   - Extracts model/benchmark information automatically
   - Creates metadata file in plugin directory
   - Stages the file for commit
3. **Note:** Files are staged but not committed here - commit happens in the unified commit job

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

**Note:** 
- This step is skipped if all plugins already have metadata files
- Generated metadata is staged for commit (committed in unified commit job)
- If generation fails for a plugin, workflow continues with other plugins
- **After commit, the mutation workflow terminates and triggers orchestrator workflow**

### 6. Layer Mapping (Conditional)

**Job:** `5. Layer Mapping` (in mutation workflow)

**When:** Only runs if:
- New models were added (`needs_mapping == true`)
- All tests passed (`all_tests_pass == true`)
- Metadata generation completed (or was skipped)

**Purpose:** Map model layers for new model submissions (vision domain only)

**Process:**
1. **For language domain:** Step is skipped via `if: env.DOMAIN != 'language'` condition
2. **For vision domain:**
   - Extracts list of new models from plugin info
   - Triggers Jenkins layer mapping job via `actions_helpers.py trigger_layer_mapping`
   - Passes model information to Jenkins:
     - Model identifiers
     - PR number
     - Source repository and branch
   - Layer mapping files are generated and staged for commit

**Jenkins Job (vision only):**
- Runs layer mapping for new models
- Maps model architecture layers
- Updates model metadata

**Note:** 
- This step is skipped if no new models are added
- **Layer mapping is automatically skipped for language domain** (only needed for vision models)
- Layer mapping files are staged for commit (committed in unified commit job)
- **After commit, the mutation workflow terminates and triggers orchestrator workflow**

### 7. Commit and Push (Mutation Workflow)

**Job:** `6. Commit and Push` (in mutation workflow)

**When:** Only runs if:
- At least one mutation job succeeded (metadata generation, metadata update, or layer mapping)

**Purpose:** Commit all mutations and push to PR branch, then terminate workflow

**Process:**
1. Checks for staged changes from previous jobs (metadata files, layer mapping files)
2. Stages any additional unstaged changes
3. Creates unified commit with all mutations
4. Pushes commit to PR branch using PAT (to trigger workflows)
5. **Workflow terminates** - relies on synchronize event to trigger orchestrator

**Note:**
- This is a terminal step - workflow ends here
- The commit triggers a `pull_request.synchronize` event
- The orchestrator workflow then runs on the updated PR

### 8. Auto-merge (Conditional)

**Job:** `3. Auto-merge` (in orchestration workflow)

**When:** Only runs if:
- PR is validated (`is_automergeable == true`)
- All tests pass (`all_tests_pass == true`)
- PR is structurally complete (metadata exists)

**Purpose:** Automatically merge approved PRs

**Process:**
1. Auto-approves the PR using `hmarr/auto-approve-action`
2. Merges PR using `plm9606/automerge_actions`:
   - GitHub submissions: merges PRs with `automerge` label
   - Web submissions: merges PRs with `automerge-web` label
   - Uses squash merge method
   - Deletes branch after merge

**Result:** PR is merged to `main` branch

### 9. Post-Merge Scoring

**Job:** `4. Post-Merge Scoring` (in orchestration workflow)

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

### 10. Failure Notification

**Job:** `5. Notify on Failure` (in orchestration workflow)

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
├── plugin_submission_mutate.yml         # Mutation workflow (commits changes, terminates)
├── plugin_submission_orchestrator.yml   # Orchestration workflow (handles downstream steps)
├── metadata_handler.yml                  # Reusable: processes metadata
└── user_notification_handler.yml        # Reusable: handles notifications

brainscore_language/submission/
└── actions_helpers.py                    # Python utilities for workflows
```

## Key Components

### Mutation Workflow

**File:** `.github/workflows/plugin_submission_mutate.yml`

**Key Features:**
- Handles all repository mutations (metadata generation, layer mapping)
- Commits changes to PR branch
- Terminates after commit (relies on synchronize event for continuation)
- Respects GitHub Actions' non-reentrancy model

### Orchestrator Workflow

**File:** `.github/workflows/plugin_submission_orchestrator.yml`

**Key Features:**
- Assumes PR is structurally complete (metadata exists)
- Handles downstream orchestration (automerge, scoring, notifications)
- Runs on synchronize events triggered by mutation workflow commits
- Sequential job execution with clear dependencies
- Conditional logic for different scenarios
- Comprehensive error handling

### Helper Script

**File:** `brainscore_language/submission/actions_helpers.py`

**Functions:**
- `validate_pr()` - Validates PR for automerge eligibility
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

**Phase 1 - Mutation Workflow:**
1. User creates PR with new model in `brainscore_language/models/newmodel/`
   - Model code is added but no `metadata.yml` file
2. Mutation workflow detects new model
3. Validates PR (tests must pass)
4. **Generates metadata.yml** for the new model (since it's missing)
   - Extracts model architecture, parameters, etc.
   - Stages metadata file
5. **Skips layer mapping** (language domain doesn't require it)
6. **Commits and pushes** metadata to PR branch
7. **Mutation workflow terminates**

**Phase 2 - Orchestration Workflow:**
8. Commit triggers `synchronize` event
9. Orchestrator workflow detects changes (metadata now exists)
10. Validates PR (full validation)
11. Auto-merges PR after validation
12. Post-merge: triggers scoring
    - New model scored on all public benchmarks
    - Results stored in database

### Scenario 1b: New Model Submission (With Metadata)

**Phase 1 - Mutation Workflow:**
1. User creates PR with new model in `brainscore_language/models/newmodel/`
   - Model code AND `metadata.yml` file are both provided
2. Mutation workflow detects new model
3. Validates PR (tests must pass)
4. **Skips metadata generation** (metadata already exists)
5. **Skips layer mapping** (language domain doesn't require it)
6. **No mutations to commit** - mutation workflow terminates immediately

**Phase 2 - Orchestration Workflow:**
7. Orchestrator workflow runs (may be triggered by PR creation or synchronize)
8. Detects changes (metadata exists)
9. Validates PR (full validation)
10. Auto-merges PR after validation
11. Post-merge: triggers scoring

### Scenario 2: New Benchmark Submission

1. User creates PR with new benchmark in `brainscore_language/benchmarks/newbenchmark/`
   - Benchmark code added (with or without metadata.yml)
2. Workflow detects new benchmark
3. Validates PR (tests must pass)
4. **Generates metadata.yml** if missing (extracts benchmark info)
5. Auto-merges PR (no layer mapping needed for benchmarks)
6. Post-merge: triggers scoring
   - All public models scored on new benchmark
   - Results stored in database

### Scenario 3: Metadata-Only Update

1. User creates PR with only `metadata.yml` changes
2. Workflow detects metadata-only change
3. Skips validation and scoring
4. Processes metadata directly
5. Updates database with new metadata
6. No scoring triggered

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

These must pass before auto-merge is allowed.

## Labels

The workflow uses labels to control behavior:

- **`automerge`** - Standard GitHub PR auto-merge
- **`automerge-web`** - Web submission auto-merge
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

On any PR, you'll see two workflow runs:

**First Run (Mutation - if mutations needed):**
```
Plugin Submission Mutate
├─ 1. Detect Changes (success)
├─ 2. Validate PR (success)
├─ 3. Update Existing Metadata (skipped)
├─ 4. Generate Metadata (success)
├─ 5. Layer Mapping (skipped - language domain)
└─ 6. Commit and Push (success)
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

**Note:** If no mutations are needed (metadata already exists), only the orchestrator workflow runs.

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

# Submission Module — Brain-Score Language

This module contains all the code that supports the plugin submission pipeline, from GitHub Actions helpers to Jenkins integration and scoring endpoints.

## Files

| File | Purpose |
|------|---------|
| `config.py` | Provides `get_database_secret()` — reads `BSC_DATABASESECRET` from environment |
| `endpoints.py` | Scoring entry points and Jenkins trigger (`call_jenkins_language`) |
| `actions_helpers.py` | CLI tool used by the GitHub Actions orchestrator workflow |
| `hardcoded_metadata.py` | Generates placeholder `metadata.yml` files for plugins missing metadata |

## `endpoints.py`

### `LanguagePlugins`
Implements `DomainPlugins` interface from `brainscore_core`. Provides `load_model`, `load_benchmark`, and `score` methods for the language domain.

### `run_scoring(args_dict)`
Called by Jenkins to run scoring. Resolves model and benchmark IDs, then calls `RunScoringEndpoint` for each model×benchmark pair.

### `call_jenkins_language(plugin_info)`
Triggers the Jenkins `core/job/score_plugins` job. This is the language-specific variant (vision uses `dev_score_plugins`).

- Accepts `plugin_info` as a JSON string or dict
- Fetches a CSRF crumb from Jenkins before triggering
- Uses POST with crumb in headers and trigger token as query parameter
- Authenticates via `JENKINS_USER` / `JENKINS_TOKEN` (basic auth)

### `send_email_to_submitter(uid, domain, pr_number, ...)`
Wrapper around `brainscore_core`'s email sender. Used for post-scoring notifications.

### Environment Variables
| Variable | Description |
|----------|-------------|
| `BSC_DATABASESECRET` | Database connection secret (required) |
| `JENKINS_USER` | Jenkins username |
| `JENKINS_TOKEN` | Jenkins API token |
| `JENKINS_TRIGGER` | Jenkins job trigger token |

## `actions_helpers.py`

CLI tool invoked by the orchestrator workflow. Run as:
```bash
python brainscore_language/submission/actions_helpers.py <command> [options]
```

### Commands

#### `validate_pr`
Polls GitHub commit statuses API to check if required tests have passed.

```bash
python actions_helpers.py validate_pr \
  --pr-number 123 \
  --pr-head abc123def \
  --token $GITHUB_TOKEN
```

**Required test contexts:**
- `Language Unittests, Plugins`
- `Language Unittests, Non-Plugins`
- `Language Integration Tests`
- `docs/readthedocs.org:brain-score-language`

**ReadTheDocs special handling:** If RTD returns null 4+ consecutive times within the first 2 minutes, it is excluded from validation (common for fork PRs where RTD doesn't report).

**Output:** JSON to stdout:
```json
{
  "is_automergeable": true,
  "all_tests_pass": true,
  "test_results": {
    "Language Unittests, Plugins": "success",
    "Language Unittests, Non-Plugins": "success",
    "Language Integration Tests": "success",
    "docs/readthedocs.org:brain-score-language": "success"
  }
}
```

#### `trigger_update_existing_metadata`
Triggers the Jenkins `update_existing_metadata` job for metadata-only PRs.

```bash
python actions_helpers.py trigger_update_existing_metadata \
  --plugin-dirs "brainscore_language/models/mymodel" \
  --plugin-type "models" \
  --domain "language" \
  --metadata-and-layer-map-b64 "<base64-encoded-json>"
```

#### `trigger_layer_mapping`
Triggers Jenkins layer mapping job. Only used for non-language domains (language always has `needs_mapping=false`).

#### `extract_email`
Resolves submitter email. For web submissions, looks up email from Brain-Score user ID via database. For non-web submissions, tries GitHub API, then public events. Falls back to `mferg@mit.edu`.

#### `send_failure_email`
Sends a failure notification email via Gmail SMTP.

```bash
python actions_helpers.py send_failure_email \
  "user@example.com" \
  "123" \
  "Validate PR - Tests failed: Language Unittests, Plugins" \
  "$GMAIL_USERNAME" \
  "$GMAIL_PASSWORD"
```

## `hardcoded_metadata.py`

Generates placeholder `metadata.yml` files for plugins that don't include one. Called by the orchestrator's Step 4 (Generate Mutations and Commit).

```bash
python brainscore_language/submission/hardcoded_metadata.py <plugin_dir> <plugin_type>
```

- For models: generates metadata with architecture, parameter counts, layer counts, etc.
- For benchmarks: generates metadata with stimulus set, data metadata, metric information fields
- All values are placeholders marked as "Temporary hardcoded metadata"

## Pipeline Flow

The submission module is used at various stages of the orchestrator pipeline:

1. **Step 2 (Validate PR)** → `actions_helpers.py validate_pr` polls test statuses
2. **Step 4 (Generate Mutations)** → `hardcoded_metadata.py` generates missing metadata
3. **Step 5 (Auto-merge)** → `actions_helpers.py validate_pr` re-checks tests for metadata-only PRs
4. **Step 6 (Post-Merge)** → `endpoints.py call_jenkins_language` triggers scoring; `actions_helpers.py trigger_update_existing_metadata` for metadata-only PRs
5. **Step 7 (Notify)** → `actions_helpers.py send_failure_email` sends failure notifications

## Environment Variables

| Variable | Required By | Description |
|----------|-------------|-------------|
| `BSC_DATABASESECRET` | `config.py`, `endpoints.py`, `actions_helpers.py` | Database connection secret |
| `JENKINS_USER` | `endpoints.py`, `actions_helpers.py` | Jenkins username |
| `JENKINS_TOKEN` | `endpoints.py`, `actions_helpers.py` | Jenkins API token |
| `JENKINS_TRIGGER` | `endpoints.py`, `actions_helpers.py` | Jenkins job trigger token |
| `GITHUB_TOKEN` | `actions_helpers.py` | GitHub API token for status checks |
| `GMAIL_USERNAME` | `actions_helpers.py` | Gmail account for notifications |
| `GMAIL_PASSWORD` | `actions_helpers.py` | Gmail app password for notifications |

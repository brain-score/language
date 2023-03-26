PRs that includes changes to subdirectories of plugin folders (/benchmarks, /data, /models, or /metrics) are subject to the following workflows:

automerge_plugin-only_prs.yml
Purpose: Automatically merge plugin-only PRs
Triggered on all PRs either by
- completion of CI checks, OR
- tagging with label
1) If PR is labeled "automerge" or "automerge-web" (all submissions via the Brain-Score website are tagged with the latter), checks if Travis tests pass. If yes, THEN
2) Checks if PR modifies any code outside plugin dirs. If no changes are made beyond subdirs of /benchmarks, /data, /models, or /metrics, the PR is automatically approved and merged.

score_new_plugins.yml
Purpose: Trigger scoring on new models and benchmarks
Triggered on all PRs upon merge to main
If the merged code includes changes to a subdir of /benchmarks or /models,
a Jenkins scoring run for the new benchmark(s) and/or model(s) is triggered.
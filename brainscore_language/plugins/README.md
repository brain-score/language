# A(n incomplete) guide to plugins

A "plugin" is a module that includes models, benchmarks, or metrics that are not part of the core Language Brain-Score codebase. These plugins are generally submitted by third parties and can contain additional dependencies.

## How to add your plugin to Language Brain-Score
While you can easily run your own models, benchmarks, and metrics by adding them to your local `plugins` directory, if you'd like to share them with the Language Brain-Score community, these instructions will help you prepare a plugin submission.

### Requirements
* Your plugin directory must include a test file named `test.py` that contains minimal but thorough tests for your plugin.
* If your plugin relies on additional dependencies beyond those included in the Language Brain-Score setup, your plugin directory must include a python `requirements.txt`. Please specify only the minimal versions required to run your plugin. 

### Submission
Now that your plugin is ready to share, there are two paths to submission:
1. Upload your plugin as a zipfile via [the Brain-Score website](brain-score.org)
2. Create a pull request in this repository

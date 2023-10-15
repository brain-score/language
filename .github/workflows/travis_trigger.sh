#!/bin/bash

GH_WORKFLOW_TRIGGER=$1
TRAVIS_PULL_REQUEST_SHA=$2
PLUGIN_ONLY=$3

curl -L -X POST \
-H "Authorization: token $GH_WORKFLOW_TRIGGER" \
-d '{"state": "success", "description": "Run automerge workflow - '$PLUGIN_ONLY'", 
  "context": "continuous-integration/travis"}' \
  "https://api.github.com/repos/brain-score/language/statuses/$TRAVIS_PULL_REQUEST_SHA"
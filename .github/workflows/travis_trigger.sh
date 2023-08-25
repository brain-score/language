#!/bin/bash

GH_WORKFLOW_TRIGGER=$1
TRAVIS_PULL_REQUEST_SHA=$2
STATUS=$3

curl -L -X POST \
-H "Authorization: token $GH_WORKFLOW_TRIGGER" \
-d '{"state": "'$STATUS'", "description": "Run automerge workflow", 
  "context": "continuous-integration/travis"}' \
  "https://api.github.com/repos/brain-score/language/statuses/$TRAVIS_PULL_REQUEST_SHA"
#!/bin/bash -ex

if [ ! -z "${CI_MERGE_REQUEST_TARGET_BRANCH_NAME+x}" -a "${CI_MERGE_REQUEST_EVENT_TYPE}" != "detached" ]; then
  echo Skipping as this is MR CI for ${CI_MERGE_REQUEST_TARGET_BRANCH_NAME} branch
  exit 0
fi

dest=`lib/petsc/bin/maint/check-merge-branch.sh`

if git diff --exit-code HEAD...${dest} -- .gitlab-ci.yml lib/petsc/conf/rules; then
    printf "Success! Using current CI settings as in gitlab-ci.yml in ${dest}!\n"
else
    printf "ERROR! Using old CI settings in gitlab-ci.yml! Please rebase to ${dest} to use current CI settings.\n"
    exit 1
fi


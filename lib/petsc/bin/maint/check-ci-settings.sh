#!/bin/bash -ex

if [ ! -z "${CI_MERGE_REQUEST_TARGET_BRANCH_NAME+x}" -a "${CI_MERGE_REQUEST_EVENT_TYPE}" != "detached" ]; then
  echo Skipping as this is MR CI for ${CI_MERGE_REQUEST_TARGET_BRANCH_NAME} branch
  exit 0
fi

git fetch --unshallow --no-tags origin +release:remotes/origin/release +master:remotes/origin/master
base_release=$(git merge-base --octopus origin/release origin/master HEAD)
base_master=$(git merge-base origin/master HEAD)
if [ ${base_release} = ${base_master} ]; then
    dest=origin/release
else
    dest=origin/master
fi
if git diff --exit-code HEAD...${dest} -- .gitlab-ci.yml lib/petsc/conf/rules; then
    printf "Success! Using current CI settings as in gitlab-ci.yml in ${dest}!\n"
else
    printf "ERROR! Using old CI settings in gitlab-ci.yml! Please rebase to ${dest} to use current CI settings.\n"
    exit 1
fi


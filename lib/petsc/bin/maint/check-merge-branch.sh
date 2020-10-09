#!/bin/bash -e

if [ ! -z ${CI_MERGE_REQUEST_TARGET_BRANCH_NAME+x} ]; then
  git fetch -q --unshallow --no-tags origin +${CI_MERGE_REQUEST_TARGET_BRANCH_NAME}:remotes/origin/${CI_MERGE_REQUEST_TARGET_BRANCH_NAME}
  echo origin/${CI_MERGE_REQUEST_TARGET_BRANCH_NAME}
  exit 0
fi

git fetch -q --unshallow --no-tags origin +release:remotes/origin/release +master:remotes/origin/master HEAD
base_release=$(git merge-base --octopus origin/release origin/master HEAD)
base_master=$(git merge-base origin/master HEAD)
if [ ${base_release} = ${base_master} ]; then
    dest=origin/release
else
    dest=origin/master
fi
echo ${dest}
exit 0

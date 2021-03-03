#!/bin/bash -e

if [ ! -z "${CI_MERGE_REQUEST_TARGET_BRANCH_NAME+x}" -a "${CI_MERGE_REQUEST_EVENT_TYPE}" != "detached" ]; then
  git fetch -q --unshallow --no-tags origin +${CI_MERGE_REQUEST_TARGET_BRANCH_NAME}:remotes/origin/${CI_MERGE_REQUEST_TARGET_BRANCH_NAME}
  echo origin/${CI_MERGE_REQUEST_TARGET_BRANCH_NAME}
  exit 0
fi

git fetch -q --unshallow --no-tags origin +release:remotes/origin/release +main:remotes/origin/main
base_release=$(git merge-base --octopus origin/release origin/main HEAD)
base_main=$(git merge-base origin/main HEAD)
if [ ${base_release} = ${base_main} ]; then
    dest=origin/release
else
    dest=origin/main
fi
echo ${dest}
exit 0

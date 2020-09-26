#!/bin/bash -e

if [ ! -z ${CI_MERGE_REQUEST_TARGET_BRANCH_NAME+x} ]; then
  echo origin/${CI_MERGE_REQUEST_TARGET_BRANCH_NAME}
  exit 0
fi

git fetch -q --unshallow --no-tags origin +maint:remotes/origin/maint +master:remotes/origin/master HEAD
base_maint=$(git merge-base --octopus origin/maint origin/master HEAD)
base_master=$(git merge-base origin/master HEAD)
if [ ${base_maint} = ${base_master} ]; then
    dest=origin/maint
else
    dest=origin/master
fi
echo ${dest}
exit 0

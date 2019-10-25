#!/bin/bash -ex

git fetch --unshallow --no-tags origin +maint:remotes/origin/maint +master:remotes/origin/master HEAD
base_maint=$(git merge-base --octopus origin/maint origin/master HEAD)
base_master=$(git merge-base origin/master HEAD)
if [ ${base_maint} = ${base_master} ]; then
    dest=origin/maint
else
    dest=origin/master
fi
if git diff --exit-code HEAD...${dest} -- .gitlab-ci.yml; then
    printf "Success! Using current CI settings as in gitlab-ci.yml in ${dest}!\n"
else
    printf "ERROR! Using old CI settings in gitlab-ci.yml! Please rebase to ${dest} to use current CI settings.\n"
    exit 1
fi


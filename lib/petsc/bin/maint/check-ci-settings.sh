#!/bin/bash -ex

if [ ! -z "${CI_MERGE_REQUEST_TARGET_BRANCH_NAME+x}" ] && [ "${CI_MERGE_REQUEST_EVENT_TYPE}" != "detached" ]; then
  echo Skipping as this is MR CI for "${CI_MERGE_REQUEST_TARGET_BRANCH_NAME}" branch
  exit 0
fi

if [ -n "${CI_MERGE_REQUEST_DIFF_BASE_SHA:-}" ]; then
  # GitLab already computed the merge base, so fetch only it and the target tip; comparing two
  # known commits needs no connecting history, which keeps the clone shallow.
  dest=origin/"${CI_MERGE_REQUEST_TARGET_BRANCH_NAME}"
  git cat-file -e "${CI_MERGE_REQUEST_DIFF_BASE_SHA}^{commit}" 2>/dev/null ||
    git fetch -q --no-tags --depth=1 origin "${CI_MERGE_REQUEST_DIFF_BASE_SHA}"
  git fetch -q --no-tags --depth=1 origin +"${CI_MERGE_REQUEST_TARGET_BRANCH_NAME}":remotes/"${dest}"
  range="${CI_MERGE_REQUEST_DIFF_BASE_SHA}..${dest}"
else
  dest=$(lib/petsc/bin/maint/check-merge-branch.sh)
  range="HEAD...${dest}"
fi

if git diff --exit-code "${range}" -- .gitlab-ci.yml lib/petsc/conf/rules; then
    printf "Success! Using current CI settings as in gitlab-ci.yml in %s!\n" "$dest"
else
    printf "ERROR! Using old CI settings in gitlab-ci.yml! Please rebase to %s to use current CI settings.\n" "$dest"
    exit 1
fi


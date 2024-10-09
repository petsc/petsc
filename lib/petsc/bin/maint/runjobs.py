#!/usr/bin/env python3

import sys
from subprocess import check_output

#  runjobs.py [-f] [job1 job2 ... jobN]
#
#  Sets a list of jobs to run upon the next push of the branch that is in a merge request.
#  If no jobs are listed then all jobs in the pipeline are run but without a need to un-pause the pipeline on the GitLab site.
#
#  -f: when commits in the local "branch" are not in sync with "origin/branch" - runjobs.py will not create a new local CI commit
#  for the specified "jobs list". Use '-f' to force the creation of this commit [and then use 'git push -f' to update this
#  branch's contents at GitLab] - if the intention is to overwrite these differences with your local files.
#  Otherwise, sync your local branch with "origin/branch" changes before running runjobs.py.
#
force = (len(sys.argv) > 1 and sys.argv[1] == '-f')
if force:
  alljobs = (sys.argv[2:] == [])
  jobs = sys.argv[2:]
else:
  alljobs = (sys.argv[1:] == [])
  jobs = sys.argv[1:]

try:
  check_output(r'git diff-index --quiet HEAD --', shell=True)
except Exception:
  print('Do not run on a repository with any uncommited changes')
  sys.exit(0)

if not force:
  try:
    check_output('git fetch', shell=True).decode('utf-8')
    branch = check_output('git rev-parse --abbrev-ref HEAD', shell=True).decode('utf-8').strip()
    base   = check_output('git merge-base ' + branch + ' remotes/origin/' + branch, shell=True).decode('utf-8').strip()
    aref   = check_output('git rev-parse ' + branch, shell=True).decode('utf-8').strip()
    bref   = check_output('git rev-parse remotes/origin/' + branch, shell=True).decode('utf-8').strip()
  except Exception:
    print('Unable to run git commits, not submitting jobs')
    sys.exit(0)
  if not aref == bref and not bref == base:
    print('Your repository is behind or has diverged from GitLab, not running jobs')
    print('If you plan to run git push with -f then use -f with this command')
    sys.exit(0)

with open('.gitlab-ci.yml','r') as fd:
  ci = fd.read()

Success_Message = 'Do a git push to start the job(s); after the pipeline finishes REMOVE commit '
File_Message = '# gitlab-ci.yml was automatically generated (by lib/petsc/bin/maint/runjobs.py) for running CI jobs: '

start = ci.find(File_Message)
if start > -1:
  start += 100
  end = ci.find('\n')
  try: commit = check_output('git rev-list -1 HEAD .gitlab-ci.yml', shell=True).decode('utf-8').strip()
  except Exception: commit = ''
  if (not alljobs and not ci[start:end] == ' all jobs' and eval(ci[start:end]) == jobs) or (alljobs and ci[start:end] == ' all jobs'):
    print(Success_Message + commit)
  else:
    print('runjobs.py was previously run (with different jobs), rerun after you REMOVE commit '+commit)
  sys.exit(0)

arches = list(jobs)
for arch in arches.copy():
  if ci.find('TEST_ARCH: arch-ci-'+arch) > -1:
    arches.remove(arch)
if arches:
  print('Could not locate job(s) '+str(arches))
  sys.exit(0)

extraJob='''
using-runjobs:
  extends: .test-basic
  stage: .pre
  tags:
    - gce-stage1
  script:
    - exit 5
  variables:
    GIT_STRATEGY: none
  allow_failure: true

'''

with open('.gitlab-ci.yml','w') as fd:
  if not alljobs: fd.write(File_Message + ' '+str(jobs)+'\n')
  else: fd.write(File_Message + 'all jobs\n')
  for a in ci.split('\n\n'):
    if a.startswith('pause-for-approval:'): continue
    if not alljobs:
      if a.startswith('# job for analyzing the coverage results '): break
      if a.find('CONFIG_OPTS: ') > -1: continue
      if a.find('petsc4py-') > -1: continue
      if a.find('check-each-commit:') > -1: continue
      test_arch =  a.find('TEST_ARCH: ')
      if test_arch > -1:
        arch = a[test_arch+19:]
        n = arch.find('\n')
        if n > -1:
          arch = arch[:n]
        if arch in jobs:
          fd.write(a+'\n\n')
      else:
        fd.write(a+'\n\n')
    else:
      fd.write(a+'\n\n')
  if not alljobs: fd.write(extraJob)

try:
  output = check_output('git add .gitlab-ci.yml', shell=True)
  if not alljobs:
    output = check_output('git commit -m"DRAFT: CI: Temporary commit, remove before merge! Runs only jobs: '+str(jobs)+'"', shell=True)
  else:
    output = check_output('git commit -m"DRAFT: CI: Temporary commit, remove before merge! Runs all jobs immediately, no unpause needed at GitLab"', shell=True)
except Exception:
  print('Unable to commit changed .gitlab-ci.yml file')
  sys.exit(0)

try: commit = check_output('git rev-list -1 HEAD .gitlab-ci.yml', shell=True).decode('utf-8').strip()
except Exception: commit = ''
print(Success_Message + commit)

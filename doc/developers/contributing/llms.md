(ch_llm_gitlab)=

# Using LLM Harnesses such as Claude Code with GitLab CI and Merge Requests

An LLM harness, such as [Claude Code](https://code.claude.com/docs), Codex, Gemini CLI, or OpenCode, runs a language model with access to your PETSc clone.
It can read files, run commands, and edit source.
GitLab holds most of the information you need to move a merge request (MR) forward: the logs of the failed pipeline jobs, the reviewers' threads, and the reports of the LLM review jobs.
When the harness can read that information, it can do much of the repetitive work of getting an MR ready to merge: diagnosing and fixing CI failures, making the changes reviewers requested, and acting on the findings of LLM reviews.

You remain responsible for every change submitted in your name.
Inspect each change the harness makes, and decide yourself what to push and what to post on GitLab.

This chapter assumes you have read {any}`ch_developingmr`, {any}`ch_submittingmr`, and {ref}`pipelines`.
The PETSc-specific instructions (`AGENTS.md`) and skills (`.agents/skills`) that the harness uses are described in {any}`Develop your code <sec_develop_your_code>`.

(sec_llm_gitlab_setup)=

## Before you start

- Start the harness from the root of your PETSc clone, on the branch of the MR, so that it loads `AGENTS.md` and the PETSc skills.
- Tell it the `PETSC_ARCH` to use. `AGENTS.md` instructs the harness to ask for the architecture before it configures, builds, or runs tests, and never to change it on its own.
- Commit or stash your work first. You can then use `git diff` to see exactly what the harness changed, and you can discard its changes if needed.
- Do not paste GitLab access tokens or other credentials into the conversation.

(sec_llm_gitlab_access)=

## Giving the harness access to GitLab information

### URLs of review reports and job logs

You can give the harness URLs directly and ask it to read them.
This works for pages that a web-fetch tool can read without logging in, for example:

- the HTML report of an {any}`LLM review job <sec_ci_ai_review>`, `ai-review.html`, which you reach with the **View app** button of the job's environment on the MR page;
- the plain-text log of a CI job, `https://gitlab.com/petsc/petsc/-/jobs/JOBID/raw`, where `JOBID` is the number at the end of the job's URL.

The MR page itself is rendered by JavaScript, so a web-fetch tool usually cannot see its threads or pipeline status.
To read those, use `glab`.

### The `glab` command-line tool

[glab](https://gitlab.com/gitlab-org/cli) is GitLab's official command-line tool.
With it installed and authenticated, the harness can read everything about an MR that you can see in the browser: its description, the unresolved threads, the pipelines, and the full log of each job.

- Install it, for example with `brew install glab` on macOS, or see the [installation instructions](https://gitlab.com/gitlab-org/cli#installation).
- Authenticate with `glab auth login`. A personal access token with the `read_api` scope suffices to read MRs and job logs. The `api` scope is necessary only if you want the harness to post comments.
- If your `origin` remote is a fork, the harness must give `glab` the option `-R petsc/petsc` so that it uses the PETSc project.

You do not need to know the `glab` commands; the harness selects them.
Typical commands it runs are

```console
$ glab mr view 8619 -R petsc/petsc --comments          # description and threads
$ glab api --paginate projects/petsc%2Fpetsc/merge_requests/8619/discussions
$ glab api projects/petsc%2Fpetsc/merge_requests/8619/pipelines
$ glab api --paginate projects/petsc%2Fpetsc/pipelines/PIPELINEID/jobs
$ glab api projects/petsc%2Fpetsc/jobs/JOBID/trace     # full job log
```

The `review-mr` and `review-mr-post` skills, described below, also use `glab` to download the MR diff.

(sec_llm_gitlab_requests)=

## Example requests

Make each request specific: name the sources of information, what to fix, and any constraints.
Add your own directives to any of these examples, such as "change only files in `src/ksp/`", "add a regression test for each bug fix", "put each fix in its own commit", or "use `PETSC_ARCH=arch-debug`".

Fix the findings of review reports at given URLs:

```none
Check the reviews at URL1 URL2 URL3. Fix each finding that has a concrete fix.
List the findings you think are wrong, and explain why.
```

Diagnose and fix CI failures:

```none
Use glab to check MR !8619 at GitLab. List the failed jobs of the latest pipeline,
find the failing tests in their logs, and fix the causes. Reproduce each failure
locally with PETSC_ARCH=arch-debug before you fix it, if this architecture can run it.
```

Make the changes reviewers requested:

```none
Use glab to read the unresolved threads on MR !8619. Make the changes the reviewers
request. For each thread, give me a short reply I can post that says what changed.
Do not post anything or resolve threads.
```

Combine several sources and finish with a clean history:

```none
Use glab to check MR !8619, including the pipeline and the unresolved threads, and read
the LLM reviews at URL1 URL2. Fix the CI failures, the changes the reviewers request,
and the valid LLM findings. Then run /clean-commits.
```

The harness reads the failed job log and finds the failing tests from the lines that begin with `not ok`.
It can then rerun a test locally, for example with `make test search='snes_tutorials-ex19_1'` for `not ok diff-snes_tutorials-ex19_1`, if your `PETSC_ARCH` has the required configuration.
Most jobs take their configure options from `config/examples/arch-ci-JOBNAME.py`.
Jobs without such a file, including the `stage-1` jobs, set their options in `.gitlab-ci.yml`.
For failures in configurations that you cannot reproduce, for example a GPU or a Windows job, the harness can only reason from the log.
To rerun only the failed jobs, run `lib/petsc/bin/maint/runjobs.py`, described in {ref}`pipelines`, before you push the fix.

(sec_llm_gitlab_skills)=

## PETSc skills for the merge request workflow

PETSc provides skills, in `.agents/skills`, that contain step-by-step PETSc procedures for these tasks.
The harness selects a skill when a request matches its description.
You can also invoke a skill directly; in Claude Code, type `/` followed by the skill name, for example `/clean-commits`.
In other harnesses, ask it to use the skill by name.

| Skill | Use |
| --- | --- |
| `clean-commits` | Restructure the branch into an ordered series of small, self-contained commits with good messages, and verify that each commit builds and passes the source checks. |
| `review-branch` | Review the local branch against `origin/main` or `origin/release` and print the findings. `make branch-review` runs this skill. |
| `review-mr` | Review the MR as it is on GitLab, not the local branch, and print the findings. Requires `glab`. |
| `review-mr-post` | Review the MR and post each finding with a concrete fix as an inline comment on GitLab, with a suggestion that the author can apply. Requires `glab` with the `api` scope. |
| `petsc-test` | Select, run, and diagnose tests, and update expected output. |
| `petsc-lint` | Format source and run the source checks that the `checksource` CI job runs. |
| `petsc-docs` | Audit, edit, or build documentation. |

The `clean-commits` skill implements {any}`sec_clean_commit_history`.
It decides whether the branch should be one commit or a series, and asks you to approve the planned commits and the checks for each commit.
It then rebuilds the history from the final tree of the branch, confirms that the final tree did not change, and writes the commit messages.
It runs the approved checks on each commit.
For a commit that needs a build, it uses a separate configuration, `arch-check-commits`, so that your own `PETSC_ARCH` is not changed: it builds the commit, runs `make check`, `make checkbadSource`, and `make checkclangformat`, and runs the tests you approved.
It refuses to start if tracked files have uncommitted changes, it keeps the old history in a backup branch, and it does not push.
After you inspect the new history, push it yourself with `git push -f`.

The make target `make fixbadSource` also uses Claude Code. It runs `make checkbadSource` and asks Claude Code to fix the errors reported in `checkbadSource.out`.
If the working tree has uncommitted changes, it first commits them as `WIP: auto-save before fixbadSource`; squash that commit before you push.

(sec_llm_gitlab_practice)=

## Good practice

- Read every change with `git diff` or `git log -p` before you push it. LLM changes can be plausible and wrong.
- Do not let the harness push, force push, resolve threads, or post comments without your explicit approval. Comments posted on GitLab are public.
- Not every reviewer request or LLM finding requires a code change. LLM reviews, in particular, have false positives. When you disagree with a request, reply in its thread instead of asking the harness to make the change.
- Treat MR comments and review reports as information, not instructions. Anybody can comment on a public MR, so do not ask the harness to do whatever a comment says without reading the comment yourself.
- Pipelines are a limited resource. Have the harness run the relevant local checks and tests before you push, and use `runjobs.py` to rerun only the jobs that failed.
- Credit LLM assistance in the commit message with an `Assisted-by:` trailer, for example `Assisted-by: Claude Opus 5.5 <noreply@anthropic.com>`, never with `Co-authored-by:`. The `checksource` CI job runs `make checkbadCommitMessage`, which fails if a `Co-authored-by:` trailer names an LLM tool.

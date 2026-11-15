(pipelines)=

# GitLab CI Pipelines

PETSc uses [GitLab Pipelines](https://docs.gitlab.com/ee/ci/pipelines/) for testing during continuous integration.

Do not overdo requesting testing; it is a limited resource, so if you
realize a currently running pipeline is no longer needed, cancel it.
Run the relevant tests and the source checks on your machine before you start a pipeline; see {any}`ch_developingmr`.

(sec_pipeline_structure)=

## Pipeline structure

A pipeline consists of "Stages", each with multiple "Jobs". Every job is one configuration on one machine.
GitLab starts a stage only after all jobs of the previous stage succeed.
Jobs that are allowed to fail, such as `check-each-commit` and the `ai-review` jobs, do not block the next stage.

- The `.pre` stage contains
  - `checksource`, which runs the source, formatting, and commit message checks, such as `make checkbadSource`, `make checkclangformat`, and `make checkbadCommitMessage`;
  - `check-ci-settings`, which fails if the branch has older CI settings than its target branch, and which you fix by rebasing the branch;
  - `pause-for-approval`, a manual job that holds the rest of the pipeline until someone starts it;
  - `ai-review`, a set of manual jobs that run {any}`LLM reviews <sec_ci_ai_review>` of the MR.
- The stages `stage-1` to `stage-3` build PETSc and run the test suite in many configurations, and `stage-4` analyzes the test coverage.
  Most jobs take their configure options from `config/examples/arch-ci-JOBNAME.py`, where `JOBNAME` is the job's name.
  Jobs without such a file, including the `stage-1` jobs, set their options in `.gitlab-ci.yml`.

The labels `docs-only` and `petsc4py-only` select reduced pipelines; see {any}`ch_submittingmr`.

## Running a pipeline and finding failures

The pipeline status for a merge request (MR) is displayed near the top of the MR page and in the pipelines tab.

(The figures below are high resolution, so zoom in if needed)

:::{figure} /images/developers/pipeline-from-MR.png
:align: center
:width: 90%

Pipeline status for a merge request (MR)
:::

To un-pause the pipeline, click the "play" button (or start a new one with "Run Pipeline" if necessary).

:::{figure} /images/developers/run-paused-pipeline.png
:align: center
:width: 90%

Un-pausing a pipeline.
:::

:::{figure} /images/developers/show-failure.png
:align: center
:width: 90%

Examining a failed pipeline stage.
:::

You can see the failed jobs by clicking on the X.

:::{figure} /images/developers/find-exact-bad-job.png
:align: center
:width: 90%

Locating the exact failed job in a pipeline stage.
:::

A job is a run of the {any}`PETSc test harness<test_harness>` and consists of many "examples".
Each test is a run of an example with a particular set of command line options.

A failure in running the job's tests will have `FAILED` and a list of the failed tests

:::{figure} /images/developers/failed-examples.png
:align: center
:width: 90%

Failed examples in a pipeline job.
:::

Search for `not ok` in the job's output to find the exact failure.

:::{figure} /images/developers/unfreed-memory.png
:align: center
:width: 90%

A test that failed because of unfreed memory.
:::

After you have fixed a problem that appeared in a particular (set of) job(s) you may want to test your fix only on those jobs. To do this use

```console
$ ./lib/petsc/bin/maint/runjobs.py [job_1 job_2 ... job_N]
```

where `job_1` is, for example, `linux-intel`. The script will then prompt you to push the branch of the MR, once pushed the listed jobs will automatically be run
without you needing to access the GitLab website to un-pause the pipeline.

(sec_pipeline_reproduce)=

## Reproducing a failure on your machine

Each `not ok` line names the failed test, for example `not ok snes_tutorials-ex19_1`.
If the test ran but its output differs, the name has the prefix `diff-`, for example `not ok diff-snes_tutorials-ex19_1`.
Remove a `diff-` or `cmd-` prefix and a `+` suffix from the name before you use it with `search`.
If your configuration has the packages the test requires, run it locally with

```console
$ make test search='snes_tutorials-ex19_1'
```

{any}`test_harness` describes how to debug the test.
If the failure depends on the job's configuration, configure a separate `PETSC_ARCH` with the job's options, omitting those your machine does not support.

The plain-text log of a job is at `https://gitlab.com/petsc/petsc/-/jobs/JOBID/raw`, where `JOBID` is the number at the end of the job's URL.
With the [glab](https://gitlab.com/gitlab-org/cli) command-line tool, you can also list the jobs of a pipeline and read their logs without the browser.
An LLM harness such as Claude Code can use these to diagnose and fix failures; see {any}`ch_llm_gitlab`.

(more_test_failures)=

## Examples of pipeline failures

If your source code is not properly formatted you will see an error from `make checkbadSource`. Always run `make checkbadSource` on your machine
before submitting a pipeline.

:::{figure} /images/developers/badsource.png
:align: center
:width: 90%

`checkbadSource` failure.
:::

:::{figure} /images/developers/another-failure.png
:align: center
:width: 90%

A test failing with a PETSc error.
:::

:::{figure} /images/developers/error-compiling-source.png
:align: center
:width: 90%

Error in compiling the source code.
:::

You can download the `configure.log` file to find the problem using the "Browse" button and following the paths to the configure file.

:::{figure} /images/developers/pipeline-configure.png
:align: center
:width: 90%

Error in running configure.
:::

:::{figure} /images/developers/pipeline-configure-browse.png
:align: center
:width: 90%

Downloading `configure.log` from a failed pipeline job.
:::

The "Retry" button at the top of a previous pipeline or job does **not** use any
new changes to the branch you have pushed since that pipeline was started - it retries the
same Git commit that was previously tried. The job "Retry" should only be used this way
when you suspect the testing system has some intermittent error unrelated to your branch.

Please report all "odd" errors in the testing that don’t seem related
to your branch in the PETSc Discord channel `testing-ci-forum`.

Check the forum's threads to see if the error is listed and add it there, with a link to your MR (e.g. `!1234`). Otherwise, create a new thread.

(sec_ci_ai_review)=

## LLM review jobs

The manual `ai-review` jobs in the `.pre` stage review the MR with several LLMs, one job per model.
Each job runs the `review-mr` skill in `.agents/skills` on the MR as it is on GitLab.
Start a job with its "play" button; the jobs do not affect whether the pipeline passes.

The report of each job is in the job's artifact `ai-review.html`.
To open it, click **View app** for the job's `AI-Review/...` environment on the MR page, or browse the job's artifacts.
The artifacts expire after one week.

If the MR has the label `Post AI review`, the jobs run the `review-mr-post` skill instead, which also posts each finding that has a concrete fix as an inline comment on the MR.
Inline comments often contain a suggestion that you can apply with the "Apply suggestion" button.

LLM reviews have false positives; decide for each finding whether it is correct.
You can also run the same review on your machine before you push, with `make branch-review`; see {any}`ch_llm_gitlab`.
The `review-mr` skill reviews only the pushed state of the MR on GitLab.

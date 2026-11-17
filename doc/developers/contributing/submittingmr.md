(ch_submittingmr)=

# Submitting a Merge Request

`git push` prints a URL to the terminal that you can use to start a merge request.
Alternatively, use [GitLab's web interface](https://docs.gitlab.com/ee/user/project/merge_requests/creating_merge_requests.html).

- The default **target** branch is `main`; if your branch started from `release`, select that as the target branch.
- If the merge request resolves an outstanding [issue](https://gitlab.com/petsc/petsc/issues),
  include a [closing pattern](https://docs.gitlab.com/ee/user/project/issues/managing_issues.html#default-closing-pattern)
  such as `Closes #123` in the MR’s description to automatically have the issue closed when the MR is merged [^closing-patterns-release] .

If you are not contributing from a fork:

- Select appropriate [labels](https://gitlab.com/petsc/petsc/-/labels) including a {any}`workflow label <sec_workflow_labels>`.
- Assign yourself to the MR.
- Select reviewers for the MR; clicking on `> Approval Rules` will list appropriate reviewers.
- If the branch started from `release` select the `milestone` of `Vxx.yy-release-fixes`
- For changes only to documentation, add the `docs-only` label, which will
  trigger a modified pipeline to build a preview of the documentation automatically.
  Any warnings from Sphinx will cause the pipeline to fail. Once completed, click "View App" on the right side in the middle of the MR page.
- For changes limited to petsc4py, add the `petsc4py-only` label to run the
  petsc4py build/test configurations, installation and pip checks, and documentation
  preview, together with the source and CI checks. The `docs-only` label takes
  precedence if both labels are present. Remove `petsc4py-only` to restore the full pipeline.
- Once the MR has been submitted, refresh the browser and select Pipelines to examine and run testing; see {ref}`pipelines`.

For MRs from forks:

- Make sure the fork is public -- as GitLab merge request process does not work well with a private fork.
- Select the correct target repository `petsc/petsc` along with the target branch.
- Select the "Allow commits from members who can merge to the target branch" option.
- Pipelines for MRs from forks run only after a developer starts them, because they use PETSc's test machines.
- GitLab does not allow you to set labels, so `@`-mention one of the developers in a comment so they can assign someone to the MR to add labels, run pipelines, and generally assist with the MR. The assignee and submitter should be listed in the upper right corner as assignees to the MR.

(sec_workflow_labels)=

## Workflow labels

The MR process, including testing and reviewing, is managed by [workflow labels](https://gitlab.com/petsc/petsc/-/labels?subscribed=&search=workflow%3A%3A) that indicate the state of the MR. Every MR should have exactly one of these labels.

The standard workflow has three steps.

- `workflow::Pipeline-Testing` The user is testing their branch. Generally, unless asked, no one else has a reason to look at such an MR.
- `workflow::Review` The user would like their branch reviewed.
- `workflow::Ready-For-Merge` The MR has passed all tests, passed the review, has no outstanding threads, and has a {any}`clean commit history <sec_clean_commit_history>`.

The submitter/assignee of the MR is responsible for changing the `workflow` label appropriately during the MR process.

Some MRs may begin with either of the following `workflow` states.

- `workflow::Request-For-Comment` The branch is not being requested to be merged, but the user would like feedback on the branch. You do not need to test the code in this state.
- `workflow::In-Development` The developer is working on the branch. Other developers not involved in the branch generally have no reason to look at these MRs.

These should also be marked as "Draft" on the MR page.
The developer usually eventually converts these two states to `workflow::Review`.

You can run the pipelines on an MR in any workflow state.

(sec_mr_reviewing)=

## MR reviewing

Once the MR has passed the pipeline, it is ready for review.
The submitter/assignee must change the {any}`workflow label <sec_workflow_labels>` to `workflow::Review`.

It is the **submitter/assignee’s** responsibility to track the progress of the MR
and ensure it gets merged.

If the pipeline detects problems, it is the **submitter/assignee’s**
responsibility to fix the errors.

Reviewers comment on the MR, either

- by clicking on the left end of a specific line in the `Changes` window. A useful feature is the ["insert suggestion"](https://docs.gitlab.com/ee/user/project/merge_requests/reviews/suggestions.html) button in the comment box, to suggest an exact replacement on a line or several adjacent lines.
- or in the overview if it is a general comment. When introducing a new topic (thread) in reviewing an MR, one should submit with "Start Review" and not "Comment".

GitLab MRs use "threads" to track discussions.
When responding to a thread, make sure to use the "Reply" box for that
thread; do not introduce a new thread or a comment.

In addition to human reviewers, developers can start LLM reviews of the MR from the pipeline; see {any}`sec_ci_ai_review`.
Treat their findings like those of any other reviewer, but expect some false positives.

{any}`sec_mr_updating` describes how to respond to the reviews and to pipeline failures.

If the **submitter/assignee** feels the MR is not getting reviewed in a timely
manner, they may assign additional reviewers to the MR and request in the MR discussion these same people to review by @-mentioning them.

When reviewers believe an MR is ready to be merged, they approve it.
You can determine who must approve your MR by clicking on the "View eligible reviewers" towards the top of the "Overview" page.

When a sufficient number of reviewers has approved the merge, the pipeline passes, new commits have been {any}`properly rearranged <sec_clean_commit_history>` if needed, and all threads have been resolved,
the **submitter/assignee** must set the label to {any}`workflow::Ready-For-Merge <sec_workflow_labels>`.

(sec_mr_updating)=

## Updating the MR in response to CI failures and reviews

Most MRs go through several rounds of changes before they are merged.
In each round, the **submitter/assignee**:

1. Reads the failed pipeline jobs ({ref}`pipelines`) and the unresolved threads.
2. Fixes the problems on the local branch, and checks the fixes locally with the tests and source checks described in {any}`ch_developingmr`.
3. Pushes the branch. If the branch history was rewritten, for example to {any}`keep the history clean <sec_clean_commit_history>`, use `git push -f`.
   To run only the jobs that failed, run `lib/petsc/bin/maint/runjobs.py` before the push, and remove its temporary commit after the pipeline finishes.
4. Replies in each thread with what was changed, or why no change is necessary, and marks the thread as resolved when the problem is fixed.
   If a reviewer suggested an exact replacement, the "Apply suggestion" button commits it directly to the branch; get that commit with `git pull --ff-only` before you change the branch locally.
5. Runs the pipeline again, unless `runjobs.py` already started the failed jobs.

After a push, the "Changes" tab of the MR can compare any two versions of the branch, which lets reviewers see only what changed since their last review.

An LLM harness such as Claude Code can read the failed job logs, the threads, and the LLM review reports, and then make many of these fixes for you; see {any}`ch_llm_gitlab`.

```{rubric} Footnotes
```

[^closing-patterns-release]: Unfortunately, these closing patterns [only work for MRs to a single default branch](https://gitlab.com/gitlab-org/gitlab/-/issues/14289) (`main`), so you must manually close related issues for MRs to `release`.

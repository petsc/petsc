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
- If appropriate, once the MR has been submitted, refresh the browser and select Pipelines to examine and run testing; see doc:`/developers/contributing/pipelines`.

For MRs from forks:

- Make sure the fork is public -- as GitLab merge request process does not work well with a private fork.
- Select the correct target repository `petsc/petsc` along with the target branch.
- Select the "Allow commits from members who can merge to the target branch" option.
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

The **submitter/assignee** must mark threads as resolved when they fix the related
problem.

Often, the submitter/assignee will need to update their branch in response to these comments,
and re-run the pipeline.

If the **submitter/assignee** feels the MR is not getting reviewed in a timely
manner, they may assign additional reviewers to the MR and request in the MR discussion these same people to review by @-mentioning them.

When reviewers believe an MR is ready to be merged, they approve it.
You can determine who must approve your MR by clicking on the "View eligible reviewers" towards the top of the "Overview" page.

When a sufficient number of reviewers has approved the merge, the pipeline passes, new commits have been {any}`properly rearranged <sec_clean_commit_history>` if needed, and all threads have been resolved,
the **submitter/assignee** must set the label to {any}`workflow::Ready-For-Merge <sec_workflow_labels>`.

```{rubric} Footnotes
```

[^closing-patterns-release]: Unfortunately, these closing patterns [only work for MRs to a single default branch](https://gitlab.com/gitlab-org/gitlab/-/issues/14289) (`main`), so you must manually close related issues for MRs to `release`.

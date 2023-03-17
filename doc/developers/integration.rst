==============================================
Getting your code and documentation into PETSc
==============================================

PETSc uses `git <https://git-scm.com/>`__, `GitLab <https://gitlab.com/petsc/petsc>`__,
and its testing system, for its source code management.
All new code in PETSc is accepted via merge requests (MRs).

By submitting code, the contributor gives irretrievable consent to the
redistribution and/or modification of the contributed source code as
described in the `PETSc open source license <https://gitlab.com/petsc/petsc/-/blob/main/CONTRIBUTING>`__.

.. _sec_integration_branches:

Integration branches
====================

.. _sec_release_branch:



``release``
-----------

The ``release`` branch contains the latest PETSc release including bug-fixes.

Bug-fixes, along with most :any:`documentation fixes <sec_docs_only_MRs>`, should start from ``release``.

.. code-block:: console

   $ git fetch
   $ git checkout -b yourname/fix-component-name origin/release


Bug-fix updates, about every month, (e.g. 3.17.1) are tagged on ``release`` (e.g. v3.17.1).

.. _sec_main_branch:


``main``
----------

The ``main`` branch contains everything in the release branch as well as new features that have passed all testing
and will be in the next release (e.g. version 3.18). Users developing software based
on recently-added features in PETSc should follow ``main``.

New feature branches :any:`should start  <sec_developing_a_new_feature>` from ``main``.

.. code-block:: console

   $ git fetch
   $ git checkout -b yourname/fix-component-name origin/main

Before filing an MR
===================

-  Read the :any:`style`.
-  :any:`Set up your git environment <sec_setup_git>`.
-  Start a new branch and make your changes. Only use a GitLab (public) fork of PETSc if you do not have
   developer (write access) to the PETSc GitLab repository.

-  If your contribution can be logically decomposed into 2 or more
   separate contributions, submit them in sequence with different
   branches and merge requests instead of all at once.
-  Include :doc:`tests </developers/testing>` which cover any changes to the source code.
-  :any:`Run the full test suite <sec_runningtests>` on your machine.

   .. code-block:: console

      $ make alltests TIMEOUT=600

-  Run the source checkers on your machine.

   .. code-block:: console

      $ make checkbadSource
      $ make clangformat
      $ make lint

-  :any:`Create a clean commit history <sec_clean_commit_history>`, ensuring that the commits on your branch present a logical picture of your new development.


Submitting an MR
================

``git push`` prints a URL that directly starts a merge request.
Alternatively, use `GitLab's web interface <https://docs.gitlab.com/ee/user/project/merge_requests/creating_merge_requests.html>`__.

- The default target branch is ``main``; if your branch started from ``release``, select that as the target branch.
- If the merge request resolves an outstanding `issue <https://gitlab.com/petsc/petsc/issues>`__,
  include a `closing pattern <https://docs.gitlab.com/ee/user/project/issues/managing_issues.html#default-closing-pattern>`__
  such as ``Closes #123`` in the MR’s description to automatically have the issue closed when the MR is merged [#closing_patterns_release]_ .

If you have developer access (that is you are not contributing from a fork):

- Select appropriate `labels <https://gitlab.com/petsc/petsc/-/labels>`__ including a :any:`workflow label <sec_workflow_labels>`.
- Assign yourself to the MR.
- Select reviewers for the MR; clicking on ``> Approval Rules`` will list appropriate reviewers.
- If the branch started from ``release`` select the ``milestone`` of ``Vxx.yy-release-fixes``
- If appropriate, once the MR has been submitted, refresh the browser and then select Pipelines to examine and run testing, see :doc:`/developers/pipelines`.

For MRs from forks:

-  Make sure the fork is not private - as gitlab merge request process does not work well (wrt pipelines, merges) with a private fork.
-  Select the correct target repository ``petsc/petsc`` along with the target branch.
-  GitLab does not allow you to set labels so  `@`-mention one of the developers in a comment so that they can assign someone to the MR to add labels, run pipelines, and generally assist with the MR. Both the submitter and the this assignee should be listed in the upper right corner as an assigned to the MR.

.. _sec_docs_only_MRs:

Docs-only MRs
-------------

For changes only to documentation, add the ``docs-only`` label, which will
trigger a modified pipeline to automatically build a preview of the documentation.
Any warnings from Sphinx will cause the pipeline to fail. Once completed, click "View App" which is to the right side in the middle of the MR page.

Documentation changes should be made to the :any:`release branch <sec_release_branch>`
in the typical case that they apply to the release version of PETSc (including changes for the website).
Changes related only to new features in the :any:`main branch <sec_main_branch>` should be applied there.

.. _sec_mr_reviewing:

MR reviewing
============

Once the MR has passed the pipeline, it is ready for review.
The submitter/assignee must change the :any:`workflow label <sec_workflow_labels>` to ``workflow::Review``.

It is the **submitter/assigner’s** responsibility to track the progress of the MR
and ensure it gets merged.

If the pipeline detects problems it is the **submitter/assignee’s**
responsibility to fix the errors.

Reviewers comment on the MR, either

- by clicking on the left end of a specific line in the changes. A useful feature is the `"insert suggestion" <https://docs.gitlab.com/ee/user/project/merge_requests/reviews/suggestions.html>`__ button in the comment box, to suggest an exact replacement on a line or several adjacent lines.
- or in the overview if it is a general comment.  When introducing a new topic (thread) in reviewing an MR, one should submit with "Start Review" and not "Comment".

Gitlab MRs use "threads" to track discussions.
When responding to a thread make sure to use the "Reply" box for that
thread; do not introduce a new thread or a comment.

The **submitter/assignee** must mark threads as resolved when they fix the related
problem.

Often, the submitter/assignee will need to update their branch in response to these comments,
and re-run the pipeline.

If the **submitter/assignee** feels the MR is not getting reviewed in a timely
manner they may assign additional reviewers to the MR and request in the discussion these same people to review by @-mentioning them.

When reviewers believe an MR is ready to be merged, they approve it.
You can determine who must approve your MR by clicking on the "View eligible reviewers" towards the top of the "Overview" page.

When the merge has been approved by a sufficient number of reviewers, the pipeline passes, new commits have been :any:`properly rearranged <sec_clean_commit_history>` if needed, and all threads have been resolved,
the **submitter/assignee** must set the label to  :any:`workflow::Ready-For-Merge <sec_workflow_labels>`.
An integrator will then merge the MR.

.. _sec_workflow_labels:

Workflow labels
===============

The MR process, including testing and reviewing, is managed by `the workflow labels <https://gitlab.com/petsc/petsc/-/labels?subscribed=&search=workflow%3A%3A>`__ that indicate the state of the MR. Every MR should have exactly one of these labels.

The standard workflow has three steps.

-  ``workflow::Pipeline-Testing`` The user is testing their branch. Generally, unless asked, no one else has a reason to look at such an MR.
-  ``workflow::Review`` The user would like their branch reviewed.
-  ``workflow::Ready-For-Merge`` The MR has passed all tests, passed the review, has no outstanding threads, and has a :any:`clean commit history <sec_clean_commit_history>`.

The submitter/assignee of the MR is responsible for changing the ``workflow`` label  appropriately during the MR process.

Some MRs may begin with either of the following ``workflow`` states.

-  ``workflow::Request-For-Comment`` The branch is not being requested to be merged but the user would like feedback on the branch. You do not need to test the code in this state.
-  ``workflow::In-Development`` The developer is working on the branch. Other developers not involved in the branch have generally no reason to look at these MRs.

Both of these should also be marked as "Draft" on the MR page.
These two states are usually eventually converted by the developer to ``workflow::Review``.

You can run the pipelines on an MR in any workflow state.

Merge request management
========================

At any given time, at least one of the :any:`sec_core_developers` is an
"MR Shepherd" and ensures that open merge
requests progress through the :any:`review process <sec_mr_reviewing>`,
examining open merge requests and taking appropriate action.

.. list-table:: MR Shepherd Checks
      :widths: 50 50
      :align: left
      :header-rows: 1

      * - MR State
        - Action
      * - Missing a :any:`workflow label <sec_workflow_labels>` and other labels
        - Add an appropriate label, or label ``workflow::Waiting-on-Submitter`` and ask the submitter/assignee to update
      * - Without an assignee
        - Assign the submitter (if the MR is from a fork also list an appropriate developer)
      * - Without reviewers
        - Assign reviewers

If MRs are inactive for too long, remind the submitter/assignee, reviewer(s), or integrator(s) of actions to take.
If the submitter/assignee must take action, change the label to ``workflow::Waiting-on-Submitter``.

.. list-table:: MR Inactivity Thresholds
      :widths: 50 50
      :align: left
      :header-rows: 1

      * - MR state
        - Inactivity threshold
      * - ``workflow:Pipeline-Testing``
        - One week
      * - ``workflow::Review``
        - One week
      * - ``workflow::Ready-for-Merge``
        - One week
      * - ``workflow::Waiting-on-Submitter``
        - One month
      * - ``workflow::Request-for-Comment``
        - One month
      * - ``workflow::Requires-Discussion``
        - One month
      * - All others
        - One year

If a submitter has been unresponsive for a year,
close the MR, label ``workflow::Inactive-closed``,
and let the submitter know that they may reopen if desired.


.. rubric:: Footnotes

.. [#closing_patterns_release] Unfortunately, these closing patterns `only work for MRs to a single default branch <https://gitlab.com/gitlab-org/gitlab/-/issues/14289>`__ (``main``), so you must manually close related issues for MRs to ``release``.

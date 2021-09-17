==============================================
Getting your code and documentation into PETSc
==============================================

PETSc uses `git <https://git-scm.com/>`__, `GitLab <https://gitlab.com/petsc/petsc>`__,
and its testing system, for its source code management.
All new code in PETSc is accepted via merge requests (MRs).

.. _sec_integration_branches:

Integration branches
====================

.. _sec_release_branch:



``release``
-----------

The ``release`` branch contains the latest PETSc release including bug-fixes.

Bug-fix branches for the release should start from ``release``.

.. code-block:: console

   $ git checkout release
   $ git pull
   $ git checkout -b yourname/fix-component-name release


Bug-fix updates, about every month or so, (e.g. 3.14.1) are tagged on ``release`` (e.g. v3.14.1).

.. _sec_main_branch:


``main``
----------

The ``main`` branch contains all features and bug-fixes that are believed to be
stable and will be in the next release (e.g. version 3.15). Users developing software based
on recently-added features in PETSc should follow ``main``.

New feature branches and bug-fixes for ``main`` :any:`should start  <sec_developing_a_new_feature>` from ``main``.

.. code-block:: console

   $ git checkout main
   $ git pull
   $ git checkout -b yourname/fix-component-name main


Contributing
============

By submitting code, the contributor gives irretrievable consent to the
redistribution and/or modification of the contributed source code as
described in the `PETSc open source license <https://gitlab.com/petsc/petsc/-/blob/main/CONTRIBUTING>`__.

Before filing an MR
-------------------

-  :any:`Set up your git environment <setup_git>`
-  Read :any:`sec_developing_a_new_feature`
-  Read the :any:`style`
-  If your contribution can be logically decomposed into 2 or more
   separate contributions, submit them in sequence with different
   branches and merge requests instead of all at once.
-  Include tests which cover any changes to the source code.
-  Run the full test suite on your machine

   .. code-block:: console

      $ make alltests TIMEOUT=600

-  Run the source checkers on your machine

   .. code-block:: console

      $ make checkbadSource
      $ make lint

-  :any:`sec_squash_excessive_commits`


Submitting an MR
----------------

``git push`` prints a URL that directly starts a merge request

.. raw:: html

   <div name="raw_1" id="thumbwrap"> <a class="thumb" href="#raw_1"><img src="../../_images/git-push-mr.png" alt=""><span><img src="../../_images/git-push-mr.png" alt=""></span></a> </div></p>
   <div class="clearfix"></div>

Alternatively, use `GitLab's web interface <https://docs.gitlab.com/ee/user/project/merge_requests/creating_merge_requests.html>`__.
For merge requests within the main PETSc repository, `click here <https://gitlab.com/petsc/petsc/-/merge_requests/new>`__.

.. raw:: html

    <div name="raw_2" id="thumbwrap"> <a class="thumb" href="#raw_2"><img src="../../_images/mr-select-branch.png" alt=""><span><img src="../../_images/mr-select-branch.png" alt=""></span></a> </div></p>
   <div class="clearfix"></div>

Select the appropriate target branch ``main`` or ``release`` (bug-fixes only).

.. raw:: html

    <div name="raw_3" id="thumbwrap"> <a class="thumb" href="#raw_3"><img src="../../_images/mr-select-target.png" alt=""><span><img src="../../_images/mr-select-target.png" alt=""></span></a> </div></p>
   <div class="clearfix"></div>

Select appropriate `labels <https://gitlab.com/petsc/petsc/-/labels>`__ including :any:`Workflow::Pipeline-Testing <workflow_labels>`. All merge requests
and issue submissions should supply appropriate labels.

.. raw:: html

    <div name="raw_4" id="thumbwrap"> <a class="thumb" href="#raw_4"><img src="../../_images/mr-select-labels.png" alt=""><span><img src="../../_images/mr-select-labels.png" alt=""></span></a> </div></p>
   <div class="clearfix"></div>

Labels are used to track related groups of activities. To receive ``notifications`` for a label (called following a label)
go to `the labels page <https://gitlab.com/petsc/petsc/-/labels>`__
and click ``Subscribe`` on the right side of the table for each label you wish to follow. 

.. raw:: html

    <div name="raw_5" id="thumbwrap"> <a class="thumb" href="#raw_6"><img src="../../_images/label-subscribe.png" alt=""><span><img src="../../_images/label-subscribe.png" alt=""></span></a> </div></p>
   <div class="clearfix"></div>

The labels include ``GPU``, ``Fortran``, ``DMNetwork``, ``bug``, ``feature``, ``enhancement``, ``ECP``, ``CI``, ``Error-handling``, ``Tao``, ``build``, ``community``, ``debugability``, and ``maintainability``.

When you subscribe to GitLab notifications it can send a great deal of email. Mail filters can use the information inside the mail to reduce and organize the notifications.

If the merge request resolves an outstanding `issue <https://gitlab.com/petsc/petsc/issues>`__, you should include a `closing
pattern <https://docs.gitlab.com/ee/user/project/issues/managing_issues.html#default-closing-pattern>`__
such as ``Fixes #123`` in the MR’s description so that issue gets
closed once the MR is merged.


Docs-only MR
^^^^^^^^^^^^

For changes **only** to documentation you may
create your merge request, add the
``docs-only`` label and you do not need to submit the MR to a pipeline.

Only small crucial documentation changes should be made to the :any:`the release branch <sec_release_branch>`
if they apply to the release version of PETSc. All others should be applied to :any:`the main branch <sec_main_branch>`.

Feedback MR
^^^^^^^^^^^

-  Select the  label  :any:`Workflow::Request-For-Comment <workflow_labels>` and make sure to select DRAFT at the top of the MR page
-  There is also a button ``Add a task list`` (next to numbered list) if
   you edit any Markdown-supporting text area. You can use this to add
   task lists to a DRAFT MR.
-  You do not need to test the code in this state

Fork MR
^^^^^^^

-  Create the MR as above from the forked repository
-  Select the correct target repository ``petsc/petsc`` (along with the target branch)
-  Assign the MR to one of the developers.
-  Fork users cannot run the pipeline or set labels.
   Hence one of the developers has to help with these processes. (If necessary - ping a developer
   in the comments section of the MR page)

Testing
-------

The PETSc continuous integration ``pipeline`` runs the entire test suite on around 60 configurations of compilers, options, and machines, it takes about 3 hours.

Pipelines can be started/controlled from the ``Pipelines`` tab
on MR page.  When a merge request is created a pipeline is create, you must manually ``un-pause`` it for the pipeline to run.

The pipeline status is displayed near the top of the MR page (and in the pipelines tab)


.. raw:: html

   <div name="raw_6" id="thumbwrap"> <a class="thumb" href="#raw_6"><img src="../../_images/pipeline-from-MR.png" alt=""><span><img src="../../_images/pipeline-from-MR.png" alt=""></span></a> </div></p>
   <div class="clearfix"></div>

.. raw:: html

   <div name="raw_7" id="thumbwrap"> <a class="thumb" href="#raw_7"><img src="../../_images/see-mr-pipelines.png" alt=""><span><img src="../../_images/see-mr-pipelines.png" alt=""></span></a> </div></p>
   <div class="clearfix"></div>

To continue the ``un-paused``  this pipeline  (or
start a new one with ``Run Pipeline`` if necessary).

.. raw:: html

   <div name="raw_8" id="thumbwrap"> <a class="thumb" href="#raw_8"><img src="../../_images/pipeline-pause-button.png" alt=""><span><img src="../../_images/pipeline-pause-button.png" alt=""></span></a> </div></p>
   <div class="clearfix"></div>

After you continue the pipeline it will display something like

.. raw:: html

   <div name="raw_9" id="thumbwrap"> <a class="thumb" href="#raw_9"><img src="../../_images/continued-pipeline.png" alt=""><span><img src="../../_images/continued-pipeline.png" alt=""></span></a> </div></p>
   <div class="clearfix"></div>


A pipeline consists of ``Stages`` each with multiple ``Jobs``, each of these is one configuration on one machine.

.. raw:: html

   <div name="raw_10" id="thumbwrap"> <a class="thumb" href="#raw_10"><img src="../../_images/show-failure.png" alt=""><span><img src="../../_images/show-failure.png" alt=""></span></a> </div></p>
   <div class="clearfix"></div>

You can see the failed jobs by clicking on the  X.


.. raw:: html

   <div name="raw_11" id="thumbwrap"> <a class="thumb" href="#raw_11"><img src="../../_images/find-exact-bad-job.png" alt=""><span><img src="../../_images/find-exact-bad-job.png" alt=""></span></a> </div></p>
   <div class="clearfix"></div>

A job consists of many "examples". Each test is a run of an example with a particular set of command line options

A failure in running the job's tests will have ``FAILED`` and a list of the failed tests

.. raw:: html

   <div name="raw_12" id="thumbwrap"> <a class="thumb" href="#raw_12"><img src="../../_images/failed-examples.png" alt=""><span><img src="../../_images/failed-examples.png" alt=""></span></a> </div></p>
   <div class="clearfix"></div>

Search for ``not ok`` in the jobs output to find the exact failure

.. raw:: html

   <div name="raw_13" id="thumbwrap"> <a class="thumb" href="#raw_13"><img src="../../_images/unfreed-memory.png" alt=""><span><img src="../../_images/unfreed-memory.png" alt=""></span></a> </div></p>
   <div class="clearfix"></div>

This test failed because the example did not free all its objects


:any:`more_test_failures`

The pipelines organization
^^^^^^^^^^^^^^^^^^^^^^^^^^

==================   =====================   =======    =======  =======================
Pre-stage            Stage 1                 Stage 2    Stage 3  Post-stage
==================   =====================   =======    =======  =======================
Basic checks         Job 1                   Job 1      Job 1    Accumulation of results

                       example 1

                         tests               Job 2      Job 2

                       example 2

                         tests               Job 3      Job 3

                     Job 2

==================   =====================   =======    =======  =======================





MR reviewing
============

Once the MR has passed the pipeline, it has been approved, all threads have been resolved,  and :any:`the excess commits squashed <sec_squash_excessive_commits>`, it is ready for review.
Change the label on the
MR page to :any:`Workflow::Review <workflow_labels>`.

It is the **submitter’s** responsibility to track the progress of the MR
and ensure it gets merged to main (or release). If the pipeline
detect problems it is the **submitter’s** responsibility to fix the
errors.

``Overview`` shows all the comments on the MR

.. raw:: html

   <div name="raw_14" id="thumbwrap"> <a class="thumb" href="#raw_14"><img src="../../_images/mr-overview.png" alt=""><span><img src="../../_images/mr-overview.png" alt=""></span></a> </div></p>
   <div class="clearfix"></div>

There are two ways (either is fine) to comment directly on the submitted source code. Use either the ``Commits`` or ``Changes`` at the top of the MR.

.. raw:: html


   <div name="raw_15" id="thumbwrap"> <a class="thumb" href="#raw_15"><img src="../../_images/changes-or-commits.png" alt=""><span><img src="../../_images/changes-or-commits.png" alt=""></span></a> </div></p>
   <div class="clearfix"></div>

Click on the left of the code to make a comment on that line of code.

.. raw:: html

   <div name="raw_16" id="thumbwrap"> <a class="thumb" href="#raw_16"><img src="../../_images/start-comment-on-code.png" alt=""><span><img src="../../_images/start-comment-on-code.png" alt=""></span></a> </div></p>
   <div class="clearfix"></div>


Write the review text and then press ``Start a Review``

.. raw:: html

   <div name="raw_17" id="thumbwrap"> <a class="thumb" href="#raw_17"><img src="../../_images/write-review-text.png" alt=""><span><img src="../../_images/write-review-text.png" alt=""></span></a> </div></p>
   <div class="clearfix"></div>

You can also provide an exact replacement for the line you would like changed

.. raw:: html

   <div name="raw_18" id="thumbwrap"> <a class="thumb" href="#raw_18"><img src="../../_images/provide-suggestion.png" alt=""><span><img src="../../_images/provide-suggestion.png" alt=""></span></a> </div></p>
   <div class="clearfix"></div>


Gitlab MRs  use ``threads`` to track discussions on MR.
This allows Gitlab and reviewers to track what threads are not yet
resolved.

.. raw:: html

   <div name="raw_19" id="thumbwrap"> <a class="thumb" href="#raw_19"><img src="../../_images/mr-thread.png" alt=""><span><img src="../../_images/mr-thread.png" alt=""></span></a> </div></p>
   <div class="clearfix"></div>

When responding to a thread make sure to use ``Reply box`` for that
thread; do not introduce a new thread or a comment.

.. raw:: html

   <div name="raw_20" id="thumbwrap"> <a class="thumb" href="#raw_20"><img src="../../_images/mr-thread-details.png" alt=""><span><img src="../../_images/mr-thread-details.png" alt=""></span></a> </div></p>
   <div class="clearfix"></div>

The **submitter** must mark threads as resolved as they fix the related
problem.


When introducing a new topic (thread) in reviewing a MR make sure you
submit with ``Start Review`` and not the ``Comment`` green button.

You can determine who must approve your MR by clicking on the ``Viewer eligible reviewers`` towards the top of the ``Overview`` page.

.. raw:: html

   <div name="raw_21" id="thumbwrap"> <a class="thumb" href="#raw_21"><img src="../../_images/button-for-approvers.png" alt=""><span><img src="../../_images/button-for-approvers.png" alt=""></span></a> </div></p>
   <div class="clearfix"></div>

.. raw:: html

   <div name="raw_22" id="thumbwrap"> <a class="thumb" href="#raw_22"><img src="../../_images/approvers.png" alt=""><span><img src="../../_images/approvers.png" alt=""></span></a> </div></p>
   <div class="clearfix"></div>


If the **submitter** feels the MR is not getting reviewed in a timely
manner they may ``Assign`` (upper right corner of the screen) to potential
reviewers and request in the discussion these same people to review by @
mentioning them.

.. raw:: html

   <div name="raw_23" id="thumbwrap"> <a class="thumb" href="#raw_23"><img src="../../_images/mr-assign.png" alt=""><span><img src="../../_images/mr-assign.png" alt=""></span></a> </div></p>
   <div class="clearfix"></div>

When the merge has been approved, the pipeline passes, the commits have been :any:`squashed <sec_squash_excessive_commits>`, and all the threads have been resolved,
the **submitter** must set the label to  :any:`Workflow::Ready-For-Merge <workflow_labels>`.

.. _workflow_labels:

Workflow labels
---------------

The MR process, including testing and reviewing, is managed by the ``Workflow`` labels that indicate the state of the MR. The standard workflow has three steps.

-  ``Workflow::Pipeline-Testing`` The user is testing their branch. Generally, unless asked, no one else has a reason to look at such an MR.
-  ``Workflow::Review`` The user would like their branch reviewed.
-  ``Workflow::Ready-For-Merge`` The MR has passed all tests, passed the review, has no outstanding threads, and has been :any:`squashed <sec_squash_excessive_commits>`.

The submitter of the MR is responsible for changing the ``workflow`` label  appropriately during the MR process.

Some MRs may begin with either of the following ``Workflow`` states.

-  ``Workflow::Request-For-Comment`` The branch is not being requested to be merged but the user would like feedback on the branch
-  ``Workflow::In-Development`` The developer is working on the branch. Other developers not involved in the branch have generally no reason to look at these MRs.

Both of these should also be marked as ``Draft`` on the MR page.
These two states are usually eventually converted by the developer to ``Workflow::Review``

You can run the pipelines on an MR in any workflow state.


.. _more_test_failures:


Examples of pipeline failures
=============================


If your source code is not properly formatted you will see an error from ``make checkbadSource``. You should always run ``make checkbadSource``` before submitting a pipeline.

.. raw:: html

   <div name="raw_24" id="thumbwrap"> <a class="thumb" href="#raw_24"><img src="../../_images/badsource.png" alt=""><span><img src="../../_images/badsource.png" alt=""></span></a> </div></p>
   <div class="clearfix"></div>

Error in compiling the source code.

.. raw:: html

   <div name="raw_25" id="thumbwrap"> <a class="thumb" href="#raw_25"><img src="../../_images/another-failure.png" alt=""><span><img src="../../_images/another-failure.png" alt=""></span></a> </div></p>
   <div class="clearfix"></div>

Error in running configure.

.. raw:: html

   <div name="raw_26" id="thumbwrap"> <a class="thumb" href="#raw_26"><img src="../../_images/error-compiling-source.png" alt=""><span><img src="../../_images/error-compiling-source.png" alt=""></span></a> </div></p>
   <div class="clearfix"></div>

You can download the ``configure.log`` file to find the problem by using the ``Browse`` button and following the paths to the configure file.


.. raw:: html

   <div name="raw_27" id="thumbwrap"> <a class="thumb" href="#raw_27"><img src="../../_images/pipeline-configure.png" alt=""><span><img src="../../_images/pipeline-configure.png" alt=""></span></a> </div></p>
   <div class="clearfix"></div>

.. raw:: html

   <div name="raw_28" id="thumbwrap"> <a class="thumb" href="#raw_28"><img src="../../_images/pipeline-configure-browse.png" alt=""><span><img src="../../_images/pipeline-configure-browse.png" alt=""></span></a> </div></p>
   <div class="clearfix"></div>

When there are failures in some jobs and a fix has been pushed, one can save time by testing only the previously
failed jobs, before running the full pipeline. To do this, ``un-pause`` a
new pipeline (do **not** retry the previous pipeline from before your most recent push), cancel
the pipeline on the pipeline page,

.. raw:: html

   <div name="raw_29" id="thumbwrap"> <a class="thumb" href="#raw_29"><img src="../../_images/cancel-pipeline.png" alt=""><span><img src="../../_images/cancel-pipeline.png" alt=""></span></a> </div></p>
   <div class="clearfix"></div>

then retry the failed jobs by using the  ``Retry``
circular button to the right of job name.

.. raw:: html

   <div name="raw_30" id="thumbwrap"> <a class="thumb" href="#raw_30"><img src="../../_images/retry-job.png" alt=""><span><img src="../../_images/retry-job.png" alt=""></span></a> </div></p>
   <div class="clearfix"></div>

It will then look like this

.. raw:: html

   <div name="raw_31" id="thumbwrap"> <a class="thumb" href="#raw_31"><img src="../../_images/started-retry-job.png" alt=""><span><img src="../../_images/started-retry-job.png" alt=""></span></a> </div></p>
   <div class="clearfix"></div>

If the selected jobs are
successful, run the rest of the pipeline by using the ``Retry``
button at the top of the pipeline

.. raw:: html

   <div name="raw_32" id="thumbwrap"> <a class="thumb" href="#raw_32"><img src="../../_images/retry-pipeline.png" alt=""><span><img src="../../_images/retry-pipeline.png" alt=""></span></a> </div></p>
   <div class="clearfix"></div>


The retry button at the top of of a previous pipeline or job does NOT use any
new changes to the branch you have pushed since that pipeline was started - it retries exactly the
same git commit that was previously tried. The job ``retry`` should only be used in this way
when you suspect the testing system has some intermittent error that is unrelated to your branch.

Please report all "odd" errors in the testing that don’t seem related
to your branch in `issue 360 <https://gitlab.com/petsc/petsc/issues/360>`__.

   1. Check the issue's threads to see if the error is listed and add
      it there, with a link to your MR (e.g. ``!1234``). Otherwise, create a new thread.
   2. Click the three dots in the top right of the thread and select
      ``Copy link``
   3. Add this link in your MR description.

Do not overdo requesting testing; it is a limited resource, so if you
realize a currently running pipeline is no longer needed, cancel it.

.. _git:

Git instructions
================

.. _setup_git:

Git Environment
---------------

-  Set your name: ``git config --global user.name  "Your Name"``
-  Set your email: ``git config --global user.email "me@example.com"``
-  Set  ``git config --global push.default simple``

Git prompt
^^^^^^^^^^

To stay oriented when working with branches, we encourage configuring
`git-prompt <https://raw.github.com/git/git/master/contrib/completion/git-prompt.sh>`__.
In the following, we will include the directory, branch name, and
PETSC_ARCH in our prompt, e.g.

.. code-block:: console

   ~/Src/petsc (main=) arch-complex
   > git checkout release
    ~/Src/petsc (release<) arch-complex

The ``<`` indicates that our copy of release is behind the repository we are
pulling from. To achieve this we have the following in our ``.profile`` (for
bash)

.. code-block:: console

   > source ~/bin/git-prompt.sh  (point this to the location of your git-prompt.sh)
   > export GIT_PS1_SHOWDIRTYSTATE=1
   > export GIT_PS1_SHOWUPSTREAM="auto"
   > export PS1='\w\[\e[1m\]\[\e[35m\]$(__git_ps1 " (%s)")\[\e[0m\] ${PETSC_ARCH}\n\$ '

Git tab completion
^^^^^^^^^^^^^^^^^^

To get tab-completion for git commands, first download and then source
`git-completion.bash <https://raw.github.com/git/git/master/contrib/completion/git-completion.bash>`__.



.. _sec_developing_a_new_feature:

Starting a new feature branch
-----------------------------

-  Obtain the PETSc source

   - If you have write access to the PETSc `GitLab <https://gitlab.com/petsc/petsc>`__ repository

     - ``git clone git@gitlab.com/petsc/petsc``  (or just use a clone you already have)

   - Otherwise

     - `Create a fork <https://gitlab.com/petsc/petsc/-/forks/new>`__ (A fork is merely your own, complete private copy of the PETSc repository on ``GitLab``)
     - You will be asked to ``Select a namespace to fork the project``, click the green ``Select`` button
     - If you already have a clone on your machine of the PETSc repository you would like to reuse

       - ``git remote set-url origin git@gitlab.com:YOURGITLABUSERNAME/petsc.git``
     - Otherwise

       - ``git clone git@gitlab.com:YOURGITLABUSERNAME/petsc.git``

-  Make sure you start from main for a new feature branch: ``git checkout main; git pull``

-  Create and switch to a new feature branch:

   ::

        git checkout -b <loginname>/<affected-package>-<short-description>

   For example, Barry’s new feature branch on removing CPP in snes/ will
   use

   ``git checkout -b barry/snes-removecpp``. Use all lowercase and no
   additional underscores.

-  Write code and tests

-  Inspect changes: ``git status``

-  Commit code:

   -  Add new files to be committed: ``git add file1 file2`` followed by

      -  Commit all files changed: ``git commit -a`` or
      -  Commit selected files: ``git commit file1 file2 file1``

-  :any:`squash any excessive commits <sec_squash_excessive_commits>`

-  Push the feature branch for testing:
   ``git push -u origin barry/snes-removecpp``

If you have long-running development of a feature branch, you will probably
fall behind the ``main`` branch.
You can move your changes to the top
of the latest ``main`` using

.. code-block:: console

    > git rebase main (while in your branch)

Quick summary of Git commands
-----------------------------

Managing branches
^^^^^^^^^^^^^^^^^

-  Switch: ``git checkout <branchname>``, for example
   ``git checkout barry/snes-removecpp``

-  Show local and remote-tracking branches: ``git branch -a``


-  Show all branches available on remote: ``git ls-remote``. Use
   ``git remote show origin`` for a complete summary.

-  Delete local branch: ``git branch -d <branchname>`` (be **careful**, you cannot get it back)

-  Delete remote branch: ``git push origin :<branchname>`` (mind the
   colon in front of the branch name) (be **careful**, you cannot get it back)

-  Show available remotes: ``git remote -v`` (you usually only have one)

-  Checkout and track a branch available on remote:
   ``git checkout -t knepley/dm-hexfem``

   If you have multiple remotes defined, use
   ``git checkout -t <remotename>/knepley/dm-hexfem``,
   e.g. ``git checkout -t origin/knepley/dm-hexfem``

-  Checkout a branch from remote, but do not track upstream changes on
   remote: ``git checkout --no-track knepley/dm-hexfem``

Reading commit logs
^^^^^^^^^^^^^^^^^^^

-  Show logs: ``git log``
-  Show logs for file or folder: ``git log [file or directory]``
-  Show changes for each log: ``git log -p`` [file or directory]
-  Show diff:

   -  Current working tree: ``git diff [file or directory]``
   -  To other commit: ``git diff <SHA1> [file or directory]``
   -  Compare version of file in two commits:
      ``git diff <SHA1> <SHA1> [file or directory]``

-  Show changes that are in main, but not yet in my current branch:

   -   ``git log ..main [file or directory]``
   -  Tabulated by author:
      ``git shortlog v3.3..main [file or directory]``

-  Showing branches:

   -  Not yet in ``main``  ``git branch --all --no-merged main``
   -  In main ``git branch --all --merged main``
   -  Remove ``--all`` to the above to not include remote tracking
      branches (work you have not interacted with yet).

-  Find where to fix a bug:

   -  Find the bad line (e.g., using a debugger)
   -  Find the commit that introduced it: ``git blame [file]``
   -  Find the branch containing that commit:
      ``git branch --contains COMMIT`` (usually one feature branch)
   -  Fix bug: ``git checkout feature-branch-name``, fix bug,
      ``git commit``

   -  Discard changes to a file which are not yet committed:
      ``git checkout file``
   -  Discard all changes to the current working tree: ``git checkout -f``


.. _sec_commit_messages:

Writing commit messages
^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: none

   ComponentName: one-line explanation of commit

   After a blank line, write a more detailed explanation of the commit.
   Many tools do not auto-wrap this part, so wrap paragraph text at a
   reasonable length. Commit messages are meant for other people to read,
   possibly months or years later, so describe the rationale for the change
   in a manner that will make sense later.

   If any interfaces have changed, the commit should fix occurrences in
   PETSc itself and the message should state its impact on users.

   If this affects any known issues, include ``fix #ISSUENUMBER`` or
   ``see #ISSUENUM`` in the message (without quotes). GitLab will create
   a link to the issue as well as a link from the issue to this commit,
   notifying anyone that was watching the issue. Feel free to link to
   mailing list discussions or [petsc-maint #NUMBER].

Formatted commit message tags:

.. code-block:: none

   We have defined several standard commit message tags you should use; this makes it easy
   to search for specific types of contributions. Multiple tags may be used
   in the same commit message.

   \spend 1h  or 30m

   * If other people contributed significantly to a commit, perhaps by
   reporting bugs or by writing an initial version of the patch,
   acknowledge them using tags at the end of the commit message.

   Reported-by: Helpful User <helpful@example.com>
   Based-on-patch-by: Original Idea <original@example.com>
   Thanks-to: Incremental Improver <improver@example.com>

   * If work is done for a particular well defined funding
   source or project you should label the commit with one
   or more of the tags

   Funded-by: My funding source
   Project: My project name

Commit message template:

.. code-block:: none

   In order to remember tags for commit messages you can create
   a file ~/git/.gitmessage containing the tags. Then on each commit
   git automatically includes these in the editor. Just remember to
   always delete the ones you do not use. For example I have

   Funded-by:
   Project:
   \spend
   Reported-by:
   Thanks-to:


Searching git on commit messages:

.. code-block:: none

   You can search the
   commit history for all contributions for a single project etc.

   * Get summary of all commits Funded by a particular source
     git log --all --grep='Funded-by: P-ECP’ --reverse [-stat or -shortstat]

   * Get the number of insertions
    git log --all --grep='Funded-by: P-ECP' --reverse --shortstat | grep changed | cut -f5 -d" " | awk '{total += $NF} END { print total }'

   * Get the number of deletions
    git log --all --grep='Funded-by: P-ECP' --reverse --shortstat | grep changed | cut -f7 -d" " | awk '{total += $NF} END { print total }'

   * Get time
    git log --all --grep='Funded-by: P-ECP' | grep Time: | cut -f2 -d":" | sed s/hours//g | sed s/hour//g |awk '{total += $NF} END { print total }'

.. _sec_squash_excessive_commits:

Squashing excessive commits
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Often a branch accumulates extra commits from bug-fixes or tiny improvements for previous commits. These changes do not belong as separate commits but
should be included in an appropriate previous commit. These commits will often break ``git bisect``.
removing these commits is called ``squashing`` and can be done several ways, the easiest is with the ``rebase`` command.

Say you have made three commits and the most recent two are fixes for the first of the three then use

.. code-block:: none

   git rebase -i HEAD~3


TODO: include images of the processes

If the branch has already been pushed this means the ``squashed`` branch you have now is not compatible with the remote copy of the branch. You must force push your changes with

.. code-block:: none

   git push -u origin +branch-name


to update the remote branch with your copy. This must be done with extreme care and only if you know someone else has not changed the  remote copy of the branch,
otherwise you will lose those changes. **Never** do a ``git pull`` after you rebase since that will bring over the old values and insert them back into the document
making a mess of the material and its history.

You can use ``git log`` to see the recent changes to your branch and help determine what commits should be ``squashed``.

It is better to ``squash`` your commits regularly than to wait until you have a large number of them to ``squash`` because you will then not know which commits need to be combined.


Further reading
^^^^^^^^^^^^^^^

-  `Tim Pope: A note about Git commit messages <http://tbaggery.com/2008/04/19/a-note-about-git-commit-messages.html>`__
-  `Junio Hamano: Fun with merges and purposes of
   branches <http://gitster.livejournal.com/42247.html>`__
-  `LWN: Rebasing and merging: some git best
   practices <http://lwn.net/Articles/328436/>`__
-  `Linus Torvalds: Merges from
   upstream <http://yarchive.net/comp/linux/git_merges_from_upstream.html>`__
-  `petsc-dev mailing
   list <http://lists.mcs.anl.gov/pipermail/petsc-dev/2013-March/011728.html>`__

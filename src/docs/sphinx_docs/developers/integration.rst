===========================
PETSc Integration Workflows
===========================

Integration branches
====================

.. _sec_master_branch:

``master``
----------

The ``master`` branch (soon to be renamed) contains all features and bug fixes that are believed to be
stable and will be in the next release (e.g. version 3.14). Users developing software based
on recently-added features in PETSc should follow ``master``.

New feature branches should start from ``master``.

.. _sec_release_branch:

``release``
-----------

The ``release`` branch provides bug-fix patches for the latest release.
Bug fixes for the release should be started here:

.. code-block:: none

   $ git checkout -b yourname/fix-component-name release

As with new features, it will be tested and later merged to
``release`` and ``master``. Bug-fix updates (e.g. 3.14.1) are tagged on ``release`` (e.g. v3.14.1).


Contributing workflows
======================

By submitting code, the contributor gives irretrievable consent to the
redistribution and/or modification of the contributed source code as
described in the `PETSc open source license <https://gitlab.com/petsc/petsc/-/blob/master/CONTRIBUTING>`__.

Before filing a merge request
-----------------------------

-  Read the :any:`style`
-  If your contribution can be logically decomposed into 2 or more
   separate contributions, submit them in sequence with different
   branches instead of all at once.
-  Include tests which cover any changes to the source code.
-  Run the full test suite on your machine -
   i.e ``make alltests TIMEOUT=600``
-  Run source checker on your machine -
   i.e ``make checkbadSource``


Submit merge request
--------------------

-  ``git push`` prints a URL that can be used to create a merge request.
   Alternatively, use `GitLab's web interface <https://gitlab.com/petsc/petsc/merge_requests/new>`__.
-  Select the correct target branch (:any:`sec_master_branch` or :any:`sec_release_branch`).
-  Select appropriate `labels <https://gitlab.com/petsc/petsc/-/labels>`__ including "workflow::Pipeline-Testing"
-  If the merge request resolves an outstanding `issue <https://gitlab.com/petsc/petsc/issues>`__), you should include a `closing
   pattern <https://docs.gitlab.com/ee/user/project/issues/managing_issues.html#default-closing-pattern>`__
   such as "Fixes #123" in the MR’s description so that issue gets
   closed once the MR is merged.

Merge request from a fork
^^^^^^^^^^^^^^^^^^^^^^^^^

-  To use the web interface option - use the fork web page, merge requests, new merge request.
-  Select the correct target repository ``petsc/petsc`` (along with the target branch)
-  Assign the MR to one of the developers.
-  Fork users lack permissions to use pipeline resources or set labels
   mentioned in the workflow below. Hence one of the developers would
   have to help with these processes. (If necessary - ping a developer
   in the comments section of the MR page)

Test using gitlab pipelines
--------------------------

-  Test pipelines can be started/controlled from the ``Pipelines`` tab
   on MR page.  When a merge request is created a pipeline is
   automatically started (with a merge with destination branch) - but
   goes into pause state.
-  To run this pipeline `un-pause` this already started pipeline (or
   start a new one if necessary).
-  The test pipeline status is displayed near the top of the MR page
   (and in the pipelines tab)

More on MR pipelines
^^^^^^^^^^^^^^^^^^^^

-  Do not overdo requesting testing; it is a limited resource, so if you
   realize a currently running test pipeline is no longer needed, cancel it.
-  When there are failures in a some jobs - and a fix is pushed for
   these failures, one can try re-testing only with the previously
   failed jobs, before running the full pipeline. To do this, start a
   new pipeline (if one is not already auto-started by the MR), cancel
   the pipeline on the pipeline page (this cancels all the jobs in the
   pipeline), now retry the selected jobs by using the little retry
   button to the right of job name. If the selected jobs are
   successful - one can run the full pipeline by using the retry
   button at the top of the page.
-  Note the retry button at the top of pipeline page does NOT use any
   new changes to the branch when it retries - it retries exactly the
   same git commit that was previously tried (and skips the already
   successful jobs).
-  Please report all "odd" errors in the testing that don’t seem related
   to your branch in `issue 360 <https://gitlab.com/petsc/petsc/issues/360>`__.

   1. Check the current current threads to see if it is listed and add
      it there, with a link to your MR (e.g. ``!1234``). Otherwise, create a new thread.
   2. Click the three dots in the top right of the thread and select
      "copy link"
   3. Add this link in your MR description.


Submit merge requests for suggestions on design, etc.
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

-  You do not need to test the code before submitting
-  Make sure to select DRAFT at the top of the MR page
-  select the additional label "workflow::Request-For-Comment"
-  There is also a button ``Add a task list`` (next to numbered list) if
   you edit any Markdown-supporting text area. You can use this to add
   task lists to a WIP MR.

Merge request review process
----------------------------

- Once the MR is tested and ready for review, change the label on the
  MR page to "workflow::Review"

It is the **submitter’s** responsibility to track the progress of the MR
and ensure it gets merged to master (or release). If the pipeline tests
detect problems it is the **submitter’s** responsibility to fix the
errors.

Gitlab merge requests (MRs) use “threads” to track discussions on MR.
This allows Gitlab and reviewers to track what threads are not yet
resolved.

-  When introducing a new topic (thread) in reviewing a MR make sure you
   submit with ``Start thread`` and not the ``Comment`` green button.
-  When responding to a thread make sure to use ``Reply box`` for that
   thread; do not introduce a new thread or a comment.

The **submitter** must mark threads as resolved as they fix the related
issue.

If the **submitter** feels the MR is not getting reviewed in a timely
manner they may Assign (upper right corner of the screen) to potential
reviewers and request in the discussion these same people to review by @
mentioning them.

When the merge has been approved (requires codeowners, integrator
approvals), all the tests work, and all the threads have been resolved
the **submitter** must set a label to "workflow::Ready-For-Merge" (can
also assign the MR to (@sbalay) if necessary)

Docs-only changes
^^^^^^^^^^^^^^^^^

To allow for small, quick changes to documentation, if you have made
**absolutely sure** that your changes only affect documentation, you may
create your merge request, add the
“workflow::Docs-Review-Merge” label, and assign to an integrator
to review and merge.

If in doubt, use the normal review process.

Remember that documentation changes should be made to the :any:`the release branch <sec_release_branch>`
if they apply to the release version of PETSc.

GitLab instructions
===================

We use labels to track related groups of activities. To follow labels
(such as GPU or DMNetwork) go to `the labels page <https://gitlab.com/petsc/petsc/-/labels>`__
and click "Subscribe" on the right side of the table. All merge requests
and issue submissions should supply appropriate labels.

Git instructions
================

Setup
-----

-  Set your name: ``git config --global user.name  "Your Name"``
-  Set your email: ``git config --global user.email "me@example.com"``
-  Do not push local branches nonexistent on upstream by default:
   ``git config --global push.default simple`` (older versions of git
   require ``git config --global push.default tracking``)

Quick summary of Git commands for PETSc developers
--------------------------------------------------

Starting and working on a new feature branch
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

-  Make sure you start from master: ``git checkout master``

-  Create and switch to a new feature branch:

   ::

        git checkout -b <loginname>/<affected-package>-<short-description>

   For example, Barry’s new feature branch on removing CPP in snes/ will
   use

   ``git checkout -b barry/snes-removecpp``. Use all lowercase and no
   additional underscores.

-  Write code

-  Inspect changes: ``git status``

-  Commit code:

   -  Commit all files changed: ``git commit -a`` or
   -  Commit selected files: ``git commit file1 file2 file1`` or
   -  Add new files to be committed: ``git add file1 file2`` followed by
      ``git commit``. Modified files can be added to a commit in the
      same way.

-  Push feature branch to the remote for review:
   ``git push -u origin barry/snes-removecpp``

   (or equivalently,
   ``git push --set-upstream origin barry/snes-removecpp``)

Switching between and handling branches
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

-  Switch: ``git checkout <branchname>``, for example
   ``git checkout barry/snes-removecpp``

-  Show local and remote-tracking branches: ``git branch -a``

-  Show available remotes: ``git remote -v``

-  Show all branches available on remote: ``git ls-remote``. Use
   ``git remote show origin`` for a complete summary.

-  Delete local branch: ``git branch -d <branchname>`` (only after merge
   to ``master`` is complete)

-  Delete remote branch: ``git push origin :<branchname>`` (mind the
   colon in front of the branch name)

-  Checkout and track a branch available on remote:
   ``git checkout -t knepley/dm-hexfem`` (if you inspect other feature
   branches, e.g. Matt’s hexfem feature branch).

   If you have multiple remotes defined, use
   ``git checkout -t <remotename>/knepley/dm-hexfem``,
   e.g. ``git checkout -t origin/knepley/dm-hexfem``

-  Checkout a branch from remote, but do not track upstream changes on
   remote: ``git checkout --no-track knepley/dm-hexfem``

Reading commit logs
^^^^^^^^^^^^^^^^^^^

-  Show logs: ``git log``
-  Show logs for file or folder: ``git log file``
-  Show changes for each log: ``git log -p`` (add file or folder name if
   required)
-  Show diff:

   -  Current working tree: ``git diff path/to/file``
   -  To other commit: ``git diff <SHA1> path/to/file``
   -  Compare version of file in two commits:
      ``git diff <SHA1> <SHA1> path/to/file``

-  Show changes that are in master, but not yet in my current branch:

   -  At any path: ``git log ..master``
   -  Only affecting a path: ``git log ..master src/dm/impls/plex/``
   -  Tabulated by author:
      ``git shortlog v3.3..master src/dm/impls/plex``

-  Showing branches:

   -  Not yet stable: ``git branch --all --no-merged master``
   -  Will be in the next release: ``git branch --all --merged master``
   -  Remove ``--all`` to the above to not include remote tracking
      branches (work you have not interacted with yet).

-  Find where to fix a bug:

   -  Find the bad line (e.g., using a debugger)
   -  Find the commit that introduced it: ``git blame path/to/file``
   -  Find the branch containing that commit:
      ``git branch --contains COMMIT`` (usually one topic branch)
   -  Fix bug: ``git checkout topic-branch-name``, fix bug,
      ``git commit``, make Merge Request, etc.

Miscellaneous
^^^^^^^^^^^^^

-  Discard changes to a file which are not yet committed:
   ``git checkout path/to/file``
-  Discard all changes to the current working tree: ``git checkout -f``
-  Forward-port local commits to the updated upstream head on master:
   ``git rebase master`` (on feature branch)
-  Delete local branch: ``git branch -D <branchname>``
-  Delete remote branch: ``git push origin :<branchname>`` (only after
   successful integration into ``master``)

Prompt
------

To stay oriented when working with branches, we encourage configuring
`git-prompt <https://raw.github.com/git/git/master/contrib/completion/git-prompt.sh>`__.
In the following, we will include the directory, branch name, and
PETSC_ARCH in our prompt, e.g.

.. code-block:: bash

   ~/Src/petsc (master=) arch-complex
   $ git checkout release
    ~/Src/petsc (release<) arch-complex

The < indicates that our copy of release is behind the repository we are
pulling from. To achieve this we have the following in our .profile (for
bash)

.. code-block:: bash

   source ~/bin/git-prompt.sh  (point this to the location of your git-prompt.sh)
   export GIT_PS1_SHOWDIRTYSTATE=1
   export GIT_PS1_SHOWUPSTREAM="auto"
   export PS1='\w\[\e[1m\]\[\e[35m\]$(__git_ps1 " (%s)")\[\e[0m\] ${PETSC_ARCH}\n\$ '

Tab completion
--------------

To get tab-completion for git commands, first download and then source
`git-completion.bash <https://raw.github.com/git/git/master/contrib/completion/git-completion.bash>`__.

.. _sec_commit_messages:

Writing commit messages
-----------------------

.. code-block:: none

   ComponentName: one-line explanation of commit

   After a blank line, write a more detailed explanation of the commit.
   Many tools do not auto-wrap this part, so wrap paragraph text at a
   reasonable length. Commit messages are meant for other people to read,
   possibly months or years later, so describe the rationale for the change
   in a manner that will make sense later.

   If any interfaces have changed, the commit should fix occurrences in
   PETSc itself and the message should state its impact on users.

   If this affects any known issues, include "fix #ISSUENUMBER" or
   "see #ISSUENUM" in the message (without quotes). GitLab will create
   a link to the issue as well as a link from the issue to this commit,
   notifying anyone that was watching the issue. Feel free to link to
   mailing list discussions or [petsc-maint #NUMBER].

Formatted tags in commit messages:

.. code-block:: none

   We have defined several standard tags you should use; this makes it easy
   to search for specific types of contributions. Multiple tags may be used
   in the same commit message.

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
   \spend 1h  or 30m

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

   Once you have started using tags it is possible to search the
   commit history for all contributions for a single project etc.

   * Get summary of all commits Funded by a particular source
     git log --all --grep='Funded-by: P-ECP’ --reverse [-stat or -shortstat]

   * Get the number of insertions
    git log --all --grep='Funded-by: P-ECP' --reverse --shortstat | grep changed | cut -f5 -d" " | awk '{total += $NF} END { print total }'

   * Get the number of deletions
    git log --all --grep='Funded-by: P-ECP' --reverse --shortstat | grep changed | cut -f7 -d" " | awk '{total += $NF} END { print total }'

   * Get time
    git log --all --grep='Funded-by: P-ECP' | grep Time: | cut -f2 -d":" | sed s/hours//g | sed s/hour//g |awk '{total += $NF} END { print total }'

Merge commits
^^^^^^^^^^^^^

Do not use ``-m 'useless merge statement'`` when performing a merge.
Instead, let ``git merge`` set up a commit message in your editor. It
will look something like this:

.. code-block:: none

   Merge branch 'master' into yourname/your-feature

   Conflicts:
     path/to/affected/file.c
     other/conflicted/paths.h

(perhaps without a Conflicts section if there are no conflicts). In your
editor, add a short description of *why* you are merging. The final
commit can look something like this:

.. code-block:: none

   Merge branch 'master' into yourname/your-feature

   Obtain symbol visibility (PETSC_INTERN), SNESSetConvergenceHistory()
   bug fix, and SNESConvergedDefault() interface change.

   Conflicts:
     path/to/affected/file.c
     other/conflicted/paths.h

It should either be to obtain a specific feature or because some major
changes affect you. When merging to an integration branch, a short summary of the
purpose of the topic branch is useful.

Further reading
^^^^^^^^^^^^^^^

http://tbaggery.com/2008/04/19/a-note-about-git-commit-messages.html

Developing new features
-----------------------

Always start new features on a fresh branch (‘topic branch’) named after
what you intend to develop. **Always branch from** ``master``:

.. code-block:: bash

   (master) $ git checkout -b yourname/purpose-of-branch
   Switched to a new branch 'yourname/purpose-of-branch'
   (yourname/purpose-of-branch) $

The naming convention for a topic branch is

.. code-block:: none

    <yourname>/<affected-package>-[<affected-package>-...]-<short description>

For example, Matt’s work on finite elements for hexahedra within dmplex
is carried out in the topic branch ``knepley/dmplex-hexfem`` or
``knepley/dmplex-petscsection-hexfem``. Don’t use spaces or underscores,
use lowercase letters only.

Now develop your feature, committing as you go. Write :any:`good commit messages <sec_commit_messages>`.
If you are familiar with
``git rebase``, it can be used at this time to edit your local history,
making its purpose as clear as possible for the reader. When your
feature is ready for review and possible integration, run

.. code-block:: bash

   (yourname/purpose-of-branch) $ git push --set-upstream origin yourname/purpose-of-branch

You can continue to work on this branch, and use ``git push`` to make
your changes visible. Only push on *your* branches.

If you have long-running development of a feature, you will probably
fall behind the master branch.
You can replay your changes on top
of the latest ``master`` using

.. code-block:: bash

   (yourname/purpose-of-branch) $ git rebase master

Checking out (tracking) a remote branch
---------------------------------------

If you wish to work on a branch that is available on the remote (shown
via ``git remote show origin``), run

.. code-block:: bash

   git checkout <branchname>

to create a local branch that will merge from the remote branch. If your
local repository is not yet aware of the new branch at the remote
repository, run ``git fetch`` and then repeat the checkout.

Merging
-------

Every branch has a purpose. Merging into branch ``branch-A`` is a
declaration that the purpose of ``branch-A`` is better served by
including those commits that were in ``branch-B``. This is achieved with
the command

.. code-block:: bash

   (branch-A) $ git merge branch-B

Topic branches do not normally contain merge commits, but it is
acceptable to merge from ``master`` or from other topic branches if your
topic depends on a feature or bug fix introduced there. When making such
a merge, use the commit message to state the reason for the merge.

For further philosophy on merges, see

-  `Junio Hamano: Fun with merges and purposes of
   branches <http://gitster.livejournal.com/42247.html>`__
-  `LWN: Rebasing and merging: some git best
   practices <http://lwn.net/Articles/328436/>`__
-  `Linus Torvalds: Merges from
   upstream <http://yarchive.net/comp/linux/git_merges_from_upstream.html>`__
-  `petsc-dev mailing
   list <http://lists.mcs.anl.gov/pipermail/petsc-dev/2013-March/011728.html>`__

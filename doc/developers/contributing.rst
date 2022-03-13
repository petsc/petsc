=====================
Contributing to PETSc
=====================

As you gain experience in building, using, and debugging with PETSc, you
may become able to contribute!

Before contributing code to PETSc, please read the :doc:`style`. You may also
be interested to read about :doc:`design`.

See :doc:`integration` for how to submit merge requests. Note that this requires
you to :any:`use Git <sec_git>`.

Once you have gained experience with developing PETSc source code and submitted merge requests, you
can become an active member of our development and push changes directly
to the petsc repository. Send mail to petsc-maint@mcs.anl.gov to
arrange it.

How-tos
=======

Some of the source code is documented to provide direct examples/templates for common
contributions, adding new implementations for solver components:

* `Add a new PC type <https://gitlab.com/petsc/petsc/-/blob/main/src/ksp/pc/impls/jacobi/jacobi.c>`__
* `Add a new KSP type <https://gitlab.com/petsc/petsc/-/blob/main/src/ksp/ksp/impls/cg/cg.c.html>`__
* `Add a new subclass of a matrix type (implementation inheritence) <https://gitlab.com/petsc/petsc/-/blob/main/src/mat/impls/aij/seq/superlu/superlu.c.html>`__

.. _sec_doc_fixes:

Documentation fixes
===================
We welcome corrections for our :doc:`documentation </developers/documentation>`.

You can supply corrections to many web pages directly by clicking "Edit this page", on the lower right corner of the page, and
making your edits, following the instructions to make a merge request, and following the :doc:`integration process </developers/integration>`.
Add the label "docs-only".

You can also submit corrections to petsc-maint@mcs.anl.gov or `post an issue <https://gitlab.com/petsc/petsc/-/issues>`__.


Browsing Source
===============

One can browse the development repositories at the following location

 https://gitlab.com/petsc/petsc

Obtaining the development version of PETSc
==========================================

`Install Git <https://git-scm.com/downloads>`__ if it is not already installed on your machine, then obtain PETSc with the following:

.. code-block:: console

  $ git clone https://gitlab.com/petsc/petsc.git
  $ cd petsc

PETSc can now be configured as specified on the
`Installation page <https://petsc.org/release/install/>`__

To update your copy of PETSc

.. code-block:: console

  $ git pull

Once updated, you will usually want to rebuild completely

.. code-block:: console

  $ make reconfigure all

This is a shorthand version of

.. code-block:: console

  $ $PETSC_DIR/$PETSC_ARCH/lib/petsc/conf/reconfigure-$PETSC_ARCH.py && make all

If you absolutely cannot use Git then you can access tarballs directly, as indicated in :ref:`other_ways_to_obtain`.

.. _sec_git:

Git instructions
================

We provide some information on common operations, here, but to contribute you are expected to know the basics of Git usage, for example from ``git help``, ``man git``, or `the Git book <https://git-scm.com/book/en/>`__.

.. _sec_setup_git:

Setting up Git
--------------

First, `set up your Git environment <https://git-scm.com/book/en/v2/Getting-Started-First-Time-Git-Setup>`__ to establish your identity.

To stay oriented when working with branches, we encourage configuring
`git-prompt <https://raw.github.com/git/git/master/contrib/completion/git-prompt.sh>`__.

To get tab-completion for Git commands, one can download and then source
`git-completion.bash <https://raw.github.com/git/git/master/contrib/completion/git-completion.bash>`__.


.. _sec_developing_a_new_feature:

Starting a new feature branch
-----------------------------

- Obtain the PETSc source.

  - If you have write access to the PETSc `GitLab repository <https://gitlab.com/petsc/petsc>`__, use ``git clone git@gitlab.com/petsc/petsc``  (or just use a clone you already have).
  - Otherwise, `Create a fork <https://gitlab.com/petsc/petsc/-/forks/new>`__ (your own copy of the PETSc repository).

    - You will be asked to "Select a namespace to fork the project"; click the green "Select" button.
    - If you already have a clone on your machine of the PETSc repository you would like to reuse

      .. code-block:: console

           $ git remote set-url origin git@gitlab.com:YOURGITLABUSERNAME/petsc.git

      and otherwise

      .. code-block:: console

          $ git clone git@gitlab.com:YOURGITLABUSERNAME/petsc.git

-  Determine the appropriate :any:`integration branch <sec_integration_branches>` to start from (usually ``main``).
-  Create and switch to a new feature branch:

   .. code-block:: console

        $ git fetch
        $ git checkout -b <loginname>/<affected-package>-<short-description> origin/main  # or origin/release

   For example, Barry’s new feature branch on removing CPP in ``snes/`` will
   use

   .. code-block:: console

     $ git checkout -b barry/snes-removecpp origin/main``

   Use all lowercase and no additional underscores in the branch name.

-  Write code and tests.

-  Inspect changes and stage code using standard Git commands, e.g.

   .. code-block:: console

      $ git status
      $ git add file1 file2
      $ git commit

-  Commit code with :any:`good commit messages <sec_commit_messages>`.

   .. code-block:: console

      $ git commit

-  :any:`Create a clean commit history <sec_clean_commit_history>`.

-  Push the feature branch to the remote repository:

   .. code-block:: console

     % git push -u origin barry/snes-removecpp

- Once the branch is ready for submission, see :doc:`/developers/integration`.

.. _sec_commit_messages:

Writing commit messages
-----------------------

.. code-block:: none

   ComponentName: one-line explanation of commit

   After a blank line, write a more detailed explanation of the commit. Many tools do not auto-wrap this part, so wrap paragraph text at a reasonable length. Commit messages are meant for other people to read, possibly months or years later, so describe the rationale for the change in a manner that will make sense later, and which will be provide helpful search terms.

   Use the imperative, e.g. "Fix bug", not "Fixed bug".

   If any interfaces have changed, the commit should fix occurrences in PETSc itself and the message should state its impact on users.

   We have defined several standard commit message tags you should use; this makes it easy to search for specific types of contributions. Multiple tags may be used in the same commit message.

   /spend 1h or 30m

   If other people contributed significantly to a commit, perhaps by reporting bugs or by writing an initial version of the patch, acknowledge them using tags at the end of the commit message.

   Reported-by: Helpful User <helpful@example.com>
   Based-on-patch-by: Original Idea <original@example.com>
   Thanks-to: Incremental Improver <improver@example.com>

   If work is done for a particular well defined funding source or project you should label the commit with one or more of the tags

   Funded-by: My funding source
   Project: My project name

.. _sec_clean_commit_history:

Creating a clean commit history
-------------------------------

Often a branch's commit history does not present a logical series of changes.
Extra commits from bug-fixes or tiny improvements may accumulate. One commit may contain multiple orthogonal changes. The order of changes may be incorrect. Branches without a clean commit history will often break ``git bisect``.

Ideally, each commit in a submitted branch will allow PETSc to build, compile, and pass its tests, while presenting a small-as-possible set of very closely related changes.
However, especially prioritize rewriting to avoid commits which change the content of previous commits, as this makes reviewing on a per-commit basis difficult.

Use different commits for:

- fixing formatting and spelling mistakes,

- fixing a bug,

- adding a new feature,

- adding another new feature.

Rewriting history can be done in `several ways <https://git-scm.com/book/en/v2/Git-Tools-Rewriting-History>`__; the easiest is often with the interactive ``rebase`` command, which allows one to combine ("squash"), rearrange, and edit commits.

It is better to clean up your commits regularly than to wait until you have a large number of them.

For example, if you have made three commits and the most recent two are fixes for the first, you could use

.. code-block:: console

   $ git rebase -i HEAD~3


If the branch has already been pushed, the rewritten branch is not compatible with the remote copy of the branch. You must force push your changes with

.. code-block:: console

   $ git push -f origin branch-name

to update the remote branch with your copy. This must be done with extreme care and only if you know someone else has not changed the remote copy of the branch,
otherwise you will lose those changes. Never do a ``git pull`` after you rebase since that will merge the old branch into your local one and create a mess [#block_ugly_pull_merge]_.

You can use ``git log`` to see the recent changes to your branch and help determine what commits should be rearranged, combined, or split.
You may also find it helpful to use an additional tool such as
`git-gui <https://git-scm.com/docs/git-gui/>`__, `lazygit <https://github.com/jesseduffield/lazygit>`__, or `various GUI tools <https://git-scm.com/downloads/guis>`__.



.. _sec_rebasing:

Rebasing your branch
--------------------

You may also want to `rebase <https://git-scm.com/book/en/v2/Git-Branching-Rebasing>`__ your branch onto to the latest version of an :any:`integration branch <sec_integration_branches>` [#rebase_not_merge_upstream]_, if the integration branch has had relevant changes since you started working on your feature branch.

.. code-block:: console

  $ git fetch origin                              # assume origin --> PETSc upstream
  $ git checkout myname/component-feature
  $ git branch myname/component-feature-backup-1  # optional
  $ git rebase origin/main                        # or origin/release

.. _other_ways_to_obtain:

Other ways to obtain PETSc
==========================

Getting a tarball of the git main branch of PETSc
---------------------------------------------------
Use the following URL: https://gitlab.com/petsc/petsc/get/main.tar.gz

This mode is useful if you are on a machine where you cannot install
Git or if it has a firewall blocking http downloads.

After the tarballs is obtained - do the following:

.. code-block:: console

   $ tar zxf petsc-petsc-CHANGESET.tar.gz
   $ mv petsc-petsc-CHANGESET petsc

To update this copy of petsc, re-download the above tarball.
The URL above gets the latest changes immediately when they are pushed to the repository.

Getting the nightly tarball of the git main branch of PETSc
-------------------------------------------------------------

The nightly tarball will be equivalent to the release
tarball - with all the documentation built. Use the following URL:

http://ftp.mcs.anl.gov/pub/petsc/petsc-main.tar.gz

To update your copy of petsc simply get a new copy of the tar file.
The tar file at the ftp site is updated once each night [around midnight
Chicago time] with the latest changes to the development version of PETSc.

.. rubric:: Footnotes

.. [#rebase_not_merge_upstream] Rebasing is generally preferable to `merging an upstream branch <http://yarchive.net/comp/linux/git_merges_from_upstream.html>`__.

.. [#block_ugly_pull_merge] You may wish to `make it impossible to perform these usually-undesired "non fast-forward" merges when pulling <https://git-scm.com/docs/git-config#Documentation/git-config.txt-pullff>`__, with ``git config --global pull.ff only``.

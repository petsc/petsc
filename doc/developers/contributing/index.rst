.. _ch_contributing:

=====================
Contributing to PETSc
=====================

As you gain experience in building, using, and debugging with PETSc, you
will be able to contribute!

Before contributing code to PETSc, please read :any:`style`. You may also
be interested to read about :any:`design`.

PETSc uses `Git <https://git-scm.com/>`__, `GitLab <https://gitlab.com/petsc/petsc>`__,
and it's testing system, for its source code management.
All new code in PETSc is accepted via merge requests (MRs).

By submitting code, the contributor gives irretrievable consent to the
redistribution and/or modification of the contributed source code as
described in the `PETSc open-source license <https://gitlab.com/petsc/petsc/-/blob/main/CONTRIBUTING>`__.

How-Tos
-------

Some of the source code is documented to provide direct examples/templates for common
contributions, adding new implementations for solver components:

* `Add a new PC type <https://gitlab.com/petsc/petsc/-/blob/main/src/ksp/pc/impls/jacobi/jacobi.c>`__
* `Add a new KSP type <https://gitlab.com/petsc/petsc/-/blob/main/src/ksp/ksp/impls/cg/cg.c.html>`__
* `Add a new subclass of a matrix type (implementation inheritence) <https://gitlab.com/petsc/petsc/-/blob/main/src/mat/impls/aij/seq/superlu/superlu.c.html>`__


.. _sec_git:
.. _sec_setup_git:

Setting up Git
--------------

We provide some information on common operations here; for more details, see ``git help``, ``man git``, or `the Git book <https://git-scm.com/book/en/>`__.

* `Install Git <https://git-scm.com/downloads>`__ if it is not already installed on your machine, then obtain PETSc with the following:

* `Set up your Git environment <https://git-scm.com/book/en/v2/Getting-Started-First-Time-Git-Setup>`__ to establish your identity.

* To stay oriented when working with branches, we encourage configuring
  `git-prompt <https://raw.github.com/git/git/master/contrib/completion/git-prompt.sh>`__.

* To get tab-completion for Git commands, one can download and then source
  `git-completion.bash <https://raw.github.com/git/git/master/contrib/completion/git-completion.bash>`__.

Obtaining the development version of PETSc
------------------------------------------

- If you have write access to the PETSc `GitLab repository <https://gitlab.com/petsc/petsc>`__, use ``git clone git@gitlab.com/petsc/petsc``
  (or use a clone you already have).

- Otherwise, `Create a fork <https://gitlab.com/petsc/petsc/-/forks/new>`__ (your own copy of the PETSc repository).

  - You will be asked to "Select a namespace to fork the project"; click the green "Select" button.

  - If you already have a clone on your machine of the PETSc repository you would like to reuse

    .. code-block:: console

         $ git remote set-url origin git@gitlab.com:YOURGITLABUSERNAME/petsc.git

  - otherwise

    .. code-block:: console

        $ git clone git@gitlab.com:YOURGITLABUSERNAME/petsc.git


PETSc can now be configured as specified on the
`Installation page <https://petsc.org/release/install/>`__

To update your copy of PETSc

.. code-block:: console

  $ git pull

Once updated, you will usually want to rebuild it completely

.. code-block:: console

  $ make reconfigure all

This is equivalent to

.. code-block:: console

  $ $PETSC_DIR/$PETSC_ARCH/lib/petsc/conf/reconfigure-$PETSC_ARCH.py && make all

.. toctree::
   :maxdepth: 1

   developingmr
   submittingmr
   pipelines

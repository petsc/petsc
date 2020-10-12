=====================
Contributing to PETSc
=====================

As you gain experience in building, using, and debugging with PETSc, you
may become able to contribute!

Before contributing code to PETSc, please read the :doc:`style`. You may also
be interested to read about :doc:`design`.

See :doc:`integration` for information on how to submit patches and pull requests to PETSc.

Once you have gained experience with developing PETSc source code, you
can become an active member of our development and push changes directly
to the petsc repository. Send mail to petsc-maint@mcs.anl.gov to
arrange it.

How-tos
=======

Some of the source code is documented to provide direct examples/templates for common
contributions, adding new implementations for solver components:

* `Add a new PC type <https://www.mcs.anl.gov/petsc/petsc-current/src/ksp/pc/impls/jacobi/jacobi.c.html>`__
* `Add a new KSP type <https://www.mcs.anl.gov/petsc/petsc-current/src/ksp/ksp/impls/cg/cg.c.html>`__
* `Add a new subclass of a matrix type (implementation inheritence) <https://www.mcs.anl.gov/petsc/petsc-current/src/mat/impls/aij/seq/superlu/superlu.c.html>`__

Browsing Source
===============

One can browse the development repositories at the following location

 https://gitlab.com/petsc/petsc

Obtaining the development version of PETSc
==========================================

`Install Git <https://git-scm.com/downloads>`__ if it is not already installed on your machine, then obtain PETSc with the following:

.. code-block:: bash

  git clone https://gitlab.com/petsc/petsc.git
  cd petsc

PETSc can now be configured in the usual way, specified on the
`Installation page <https://www.mcs.anl.gov/petsc/documentation/installation.html>`__

To update your copy of PETSc

.. code-block:: bash

  git pull

Once updated, you will usually want to rebuild completely

.. code-block:: bash

  make reconfigure all

This is a shorthand version of

.. code-block:: bash

  ./$PETSC_ARCH/lib/petsc/conf/reconfigure-$PETSC_ARCH.py && make all

For additional help use

* ``git help`` or ``man git``
* `The Pro Git book <https://git-scm.com/book/en/>`__

If you absolutely cannot use git then you can access tarballs directly, as in :ref:`other_ways_to_obtain`.

.. _other_ways_to_obtain:

Other ways to obtain PETSc
==========================

Getting a Tarball of the git master branch of PETSc
---------------------------------------------------
Use the following URL: https://gitlab.com/petsc/petsc/get/master.tar.gz

This mode is useful if you are on a machine where you cannot install
Git or if it has a firewall blocking http downloads.

After the tarballs is obtained - do the following:

.. code-block:: bash

        tar zxf petsc-petsc-CHANGESET.tar.gz
        mv petsc-petsc-CHANGESET petsc

To update this copy of petsc, re-download the above tarball.
The URL above gets the latest changes immediately when they are pushed to the repository.

Getting the Nightly tarball of the git master branch of PETSc
-------------------------------------------------------------

The nightly tarball will be equivalent to the release
tarball - with all the documentation built. Use the following URL:

http://ftp.mcs.anl.gov/pub/petsc/petsc-master.tar.gz

To update your copy of petsc simply get a new copy of the tar file.
The tar file at the ftp site is updated once each night [around midnight
Chicago time] with the latest changes to the development version of PETSc.

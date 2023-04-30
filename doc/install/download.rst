.. _doc_download:

========
Download
========


Recommended: Obtain Release Version With Git
============================================

Use ``release`` branch from PETSc git repository - it provides latest release with additional fixes.

.. code-block:: console

   $ git clone -b release https://gitlab.com/petsc/petsc.git petsc
   $ git pull # obtain new release fixes (since a prior clone or pull)

To anchor to a release version (without intermediate fixes), use:

.. code-block:: console

   $ git checkout vMAJOR.MINOR.PATCH

We recommend users join the official PETSc :ref:`mailing lists <doc_mail>` to be submit
any questions they may have directly to the development team, be notified of new major
releases and bug-fixes, or to simply keep up to date with the current state of the
library.

Alternative: Obtain Release Version with Tarball
================================================

Tarball which contains only the source. Documentation available `online <https://petsc.org/release>`__.

- `petsc-3.19.1.tar.gz <https://ftp.mcs.anl.gov/pub/petsc/release-snapshots/petsc-3.19.1.tar.gz>`__

Tarball which includes all documentation, recommended for offline use.

- `petsc-with-docs-3.19.1.tar.gz <https://ftp.mcs.anl.gov/pub/petsc/release-snapshots/petsc-with-docs-3.19.1.tar.gz>`__


Tarball to enable a separate installation of petsc4py.

- `petsc4py-3.19.1.tar.gz  <https://ftp.mcs.anl.gov/pub/petsc/release-snapshots/petsc4py-3.19.1.tar.gz>`__

To extract the sources use:

.. code-block:: console

   $ tar xf petsc-<version number>.tar.gz

Current and older release tarballs are available at:

- `Primary server <https://ftp.mcs.anl.gov/pub/petsc/release-snapshots/>`__

- `Mirror <https://www.mcs.anl.gov/petsc/mirror/release-snapshots/>`__

.. Note::

   Older release tarballs of PETSc should only be used for
   applications that have not been updated to the latest release. We urge you, whenever
   possible, to upgrade to the latest version of PETSc.

Advanced: Obtain PETSc Development Version With Git
===================================================

Improvements and new features get added to ``main`` branch of PETSc git repository. To obtain development sources, use:

.. code-block:: console

   $ git clone https://gitlab.com/petsc/petsc.git petsc

or if you already have a local clone of petsc git repository

.. code-block:: console

   $ git checkout main
   $ git pull

More details on contributing to PETSc development are at :any:`ch_contributing`. The development version of
the documentation, which is largely the same as the release documentation is `available <https://petsc.org/main>`__.


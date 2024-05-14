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

- `petsc-3.21.1.tar.gz <https://web.cels.anl.gov/projects/petsc/download/release-snapshots/petsc-3.21.1.tar.gz>`__

Tarball which includes all documentation, recommended for offline use.

- `petsc-with-docs-3.21.1.tar.gz <https://web.cels.anl.gov/projects/petsc/download/release-snapshots/petsc-with-docs-3.21.1.tar.gz>`__


Tarball to enable a separate installation of petsc4py.

- `petsc4py-3.21.1.tar.gz  <https://web.cels.anl.gov/projects/petsc/download/release-snapshots/petsc4py-3.21.1.tar.gz>`__

To extract the sources use:

.. code-block:: console

   $ tar xf petsc-<version number>.tar.gz

Current and older release tarballs are available at:

- `Primary server <https://web.cels.anl.gov/projects/petsc/download/release-snapshots/>`__

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


Release Schedule
================

We intend to provide new releases every 6 months, and patch updates to current release every month.

New releases (for example: 3.20.0, 3.21.0, 3.22.0, etc.):

- March (end of the month)
- September (end of the month)

New patch updates (for example: 3.21.1, 2.21.2, 3.21.3, etc.):

- Last week of every month (or first week on next month - if delayed)

And with a new release of PETSc the old version will no longer get patch updates. I.e., when 3.22.0 is released, bug fixes
and any updates will go to 3.22.x - and petsc-3.21, petsc-3.20, etc., will not get any additional patch updates.

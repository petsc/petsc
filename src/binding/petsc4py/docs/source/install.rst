Installation
============
.. _petsc4py_install:


Install from PyPI using **pip**
-------------------------------

You can use :program:`pip` to install :mod:`petsc4py` and its
dependencies (:mod:`mpi4py` is optional but highly recommended)::

  $ python -m pip install mpi4py petsc petsc4py

Install from the PETSc source tree
----------------------------------

First `build PETSc <petsc:doc_install>`. Next :file:`cd` to the top of the
PETSc source tree and set the `PETSC_DIR <petsc:doc_multi>` and `PETSC_ARCH
<petsc:doc_multi>` environment variables. Run::

  $ python -m pip install src/binding/petsc4py

If you are cross-compiling, and the :mod:`numpy` module cannot be loaded on
your build host, then before invoking :file:`pip`, set the
:envvar:`NUMPY_INCLUDE` environment variable to the path that would be returned
by :samp:`import numpy; numpy.get_include()`::

  $ export NUMPY_INCLUDE=/usr/lib/pythonX/site-packages/numpy/core/include

Building the documentation
--------------------------

Install the documentation dependencies using the ``[doc]`` extra::

  $ python -m pip install "src/binding/petsc4py[doc]"

Then::

  $ cd src/binding/petsc4py/docs/source
  $ make html

The resulting HTML files will be in :file:`_build/html`.

.. note::

  Building the documentation requires Python 3.11 or later.

.. note::

  All new code must include documentation in accordance with the `documentation
  standard <documentation_standards>`
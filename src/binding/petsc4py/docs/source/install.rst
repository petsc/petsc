Installation
============
.. _petsc4py_install:


Install from PyPI using **pip**
-------------------------------

You can use :program:`pip` to install :mod:`petsc4py` and its dependencies::

  $ python -m pip install petsc petsc4py

Install from the PETSc source tree
----------------------------------

First `build PETSc <petsc:doc_install>`. Next :file:`cd` to the top of the
PETSc source tree and set the `PETSC_DIR <petsc:doc_multi>` and `PETSC_ARCH
<petsc:doc_multi>` environment variables. Run::

  $ python -m pip install src/binding/petsc4py

The installation of :mod:`petsc4py` supports multiple `PETSC_ARCH
<petsc:doc_multi>` in the form of colon separated list::

  $ PETSC_ARCH='arch-0:...:arch-N' python -m pip install src/binding/petsc4py

If you are cross-compiling, and the :mod:`numpy` module cannot be loaded on
your build host, then before invoking :program:`pip`, set the
:envvar:`NUMPY_INCLUDE` environment variable to the path that would be returned
by :samp:`import numpy; numpy.get_include()`::

  $ export NUMPY_INCLUDE=/usr/lib/pythonX/site-packages/numpy/core/include

Running the testing suite
-------------------------

When installing from source, the petsc4py complete testsuite can be run as::

  $ cd src/binding/petsc4py
  $ python test/runtests.py

or via the makefile rule ``test``::

  $ make test -C src/binding/petsc4py

Specific tests can be run using the command-line option ``-k``, e.g.::

  $ python test/runtests.py -k test_optdb

to run all the tests provided in :file:`tests/test_optdb.py`.

For other command-line options, run::

  $ python test/runtests.py --help

If not otherwise specified, all tests will be run in sequential mode.
To run all the tests with the same number of MPI processes, for example
``4``, run::

  $ mpiexec -n 4 python test/runtests.py

or::

  $ make test-4 -C src/binding/petsc4py

Building the documentation
--------------------------

Install the documentation dependencies::

  $ python -m pip install -r src/binding/petsc4py/conf/requirements-docs.txt

Then::

  $ cd src/binding/petsc4py/docs/source
  $ make html

The resulting HTML files will be in :file:`_build/html`.

.. note::

  Building the documentation requires Python 3.11 or later.

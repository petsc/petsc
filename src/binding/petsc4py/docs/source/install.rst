Installation
============
.. _petsc4py_install:


Using **pip**
-------------

You can use :program:`pip` to install :mod:`petsc4py` and its
dependencies (:mod:`mpi4py` is optional but highly recommended)::

  $ python -m pip install [--user] numpy mpi4py  (or pip install [--user] numpy mpi4py)
  $ python -m pip install [--user] petsc petsc4py (or pip install [--user] petsc petsc4py)


From PETSc source
-----------------

If you already have downloaded PETSc source and have installed the dependencies
of `petsc4py`, then to build the `petsc4py` module along with PETSc, add the
`--with-petsc4py=1` argument to the configure command when building PETSc:

  $ ./configure --with-petsc4py=1
  $ make
  $ make install

This will install PETSc and the `petsc4py` module into the PETSc directory
under the prefix specified to the PETSc configure command.

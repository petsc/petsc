Installation
============
.. _petsc4py_install:


Using **pip**
-------------

You can use :program:`pip` to install :mod:`petsc4py` and its
dependencies (:mod:`mpi4py` is optional but highly recommended)::

  $ python -m pip install [--user] numpy mpi4py  (or pip install [--user] numpy mpi4py)
  $ python -m pip install [--user] petsc petsc4py (or pip install [--user] petsc petsc4py)

If you already have downloaded PETSc simply add the option --download-petsc4py to the ./configure command you use for building PETSc.

Note that though the command has --download it doesn't actually download anything, it uses the source
in the directory src/binding/petsc4py.



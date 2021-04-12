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

If you wish to make the module importable without having to set the
`PYTHONPATH` environment variable, you may add a shortcut to the system-wide
`site-packages` directory creating a special `.pth` file with exactly one line
of Python code. This can be done by the following command, where the
system-wide path is assumed to be `/usr/lib/pythonX/site-packages` (replace `X`
with your python version):

  $ echo \
    "import sys, os;" \
    "p = os.getenv('PETSC_DIR');" \
    "a = os.getenv('PETSC_ARCH') or '';" \
    "p = p and os.path.join(p, a, 'lib');" \
    "p and (p in sys.path or sys.path.append(p))" \
    > /usr/lib/pythonX/site-packages/petsc4py.pth

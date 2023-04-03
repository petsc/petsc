PETSc for Python
================

Python bindings for PETSc.

Install
-------

If you have a working MPI implementation and the ``mpicc`` compiler
wrapper is on your search path, it is highly recommended to install
``mpi4py`` first::

  $ pip install mpi4py

Ensure you have NumPy installed::

  $ pip install numpy

and finally::

  $ pip install petsc petsc4py


Citations
---------

If PETSc for Python been significant to a project that leads to an
academic publication, please acknowledge that fact by citing the
project.

* L. Dalcin, P. Kler, R. Paz, and A. Cosimo,
  *Parallel Distributed Computing using Python*,
  Advances in Water Resources, 34(9):1124-1139, 2011.
  http://dx.doi.org/10.1016/j.advwatres.2011.04.013

* S. Balay, S. Abhyankar, M. Adams, S. Benson, J. Brown,
  P. Brune, K. Buschelman, E. Constantinescu, L. Dalcin, A. Dener,
  V. Eijkhout, J. Faibussowitsch, W. Gropp, V. Hapla, T. Isaac, P. Jolivet,
  D. Karpeyev, D. Kaushik, M. Knepley, F. Kong, S. Kruger,
  D. May, L. Curfman McInnes, R. Mills, L. Mitchell, T. Munson,
  J. Roman, K. Rupp, P. Sanan, J Sarich, B. Smith,
  S. Zampini, H. Zhang, and H. Zhang, J. Zhang,
  *PETSc/TAO Users Manual*, ANL-21/39 - Revision 3.19, 2023.
  https://petsc.org/release/docs/manual/manual.pdf

PETSc Users Manual
==================

.. important::

    The manual is currently being migrated here.
    If you are a user,
    see the current manual `here <https://www.mcs.anl.gov/petsc/petsc-current/docs/manual.pdf>`__.
    If you are a PETSc contributor, please help with the migration!

.. include:: temp_edit_needed_banner.inc

--------------

|
| **Mathematics and Computer Science Division**


--------------

| Prepared by
| **S. Balay\ 1, S. Abhyankar\ 2, M. Adams\ 3, J. Brown\ 1, P. Brune\ 1,
  K. Buschelman\ 1, L. Dalcin\ 4, A. Dener\ 1, V. Eijkhout\ 6, W.
  Gropp\ 1, D. Karpeyev\ 1, D. Kaushik\ 1, M. Knepley\ 1, D. May\ 7, L.
  Curfman McInnes\ 1, R. Mills\ 1, T. Munson\ 1, K. Rupp\ 1, P.
  Sanan\ 8, B. Smith\ 1, S. Zampini\ 4, H. Zhang\ 5, and H. Zhang\ 1**
| :sup:`1`\ Mathematics and Computer Science Division, Argonne National
  Laboratory
| :sup:`2`\ Electricity Infrastructure and Buildings Division, Pacific
  Northwest National Laboratory
| :sup:`3`\ Computational Research, Lawrence Berkeley National
  Laboratory
| :sup:`4`\ Extreme Computing Research Center, King Abdullah University
  of Science and Technology
| :sup:`5`\ Computer Science Department, Illinois Institute of
  Technology
| :sup:`6`\ Texas Advanced Computing Center, University of Texas at
  Austin
| :sup:`7`\ Department of Earth Sciences, University of Oxford
| :sup:`8`\ Institute of Geophysics, ETH Zurich
| March 2020

| This work was supported by the Office of Advanced Scientific Computing
  Research,
| Office of Science, U.S. Department of Energy, under Contract
  DE-AC02-06CH11357.

This manual describes the use of PETSc for the numerical solution of
partial differential equations and related problems on high-performance
computers. The Portable, Extensible Toolkit for Scientific Computation
(PETSc) is a suite of data structures and routines that provide the
building blocks for the implementation of large-scale application codes
on parallel (and serial) computers. PETSc uses the MPI standard for all
message-passing communication.

PETSc includes an expanding suite of parallel linear solvers, nonlinear
solvers, and time integrators that may be used in application codes
written in Fortran, C, C++, and Python (via petsc4py; see page ). PETSc
provides many of the mechanisms needed within parallel application
codes, such as parallel matrix and vector assembly routines. The library
is organized hierarchically, enabling users to employ the level of
abstraction that is most appropriate for a particular problem. By using
techniques of object-oriented programming, PETSc provides enormous
flexibility for users.

PETSc is a sophisticated set of software tools; as such, for some users
it initially has a much steeper learning curve than a simple subroutine
library. In particular, for individuals without some computer science
background, experience programming in C, C++, python, or Fortran and
experience using a debugger such as ``gdb`` or ``dbx``, it may require a
significant amount of time to take full advantage of the features that
enable efficient software use. However, the power of the PETSc design
and the algorithms it incorporates may make the efficient implementation
of many application codes simpler than “rolling them” yourself.

-  For many tasks a package such as MATLAB is often the best tool; PETSc
   is not intended for the classes of problems for which effective
   MATLAB code can be written.

-  There are several packages (listed on https://www.mcs.anl.gov/petsc),
   built on PETSc, that may satisfy your needs without requiring
   directly using PETSc. We recommend reviewing these packages
   functionality before using PETSc.

-  PETSc should *not* be used to attempt to provide a “parallel linear
   solver” in an otherwise sequential code. Certainly all parts of a
   previously sequential code need not be parallelized but the matrix
   generation portion must be parallelized to expect any kind of
   reasonable performance. Do not expect to generate your matrix
   sequentially and then “use PETSc” to solve the linear system in
   parallel.

Since PETSc is under continued development, small changes in usage and
calling sequences of routines will occur. PETSc is supported; see
https://www.mcs.anl.gov/petsc/miscellaneous/mailing-lists.html for
information on contacting support.

A list of publications and web sites that feature work involving PETSc
may be found at https://www.mcs.anl.gov/petsc/publications/.

We welcome any reports of corrections for this document at
``petsc-maint@mcs.anl.gov``.

:math:`\bullet`

Manual pages and example usage :
https://www.mcs.anl.gov/petsc/documentation/

Installing PETSc :
https://www.mcs.anl.gov/petsc/documentation/installation.html

Tutorials :
https://www.mcs.anl.gov/petsc/documentation/tutorials/index.html


.. toctree::
   :maxdepth: 2

   introduction
   programming
   additional

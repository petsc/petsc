About This Manual
-----------------

This manual describes the use of the Portable, Extensible Toolkit for Scientific Computation
(PETSc) for the numerical solution of
partial differential equations and related problems on high-performance
computers. PETSc is a suite of data structures and routines that provide the
building blocks for the implementation of large-scale application codes
on parallel (and serial) computers. PETSc uses the MPI standard for all
distributed memory communication.

PETSc includes a large suite of parallel linear solvers, nonlinear
solvers, and time integrators that may be used in application codes
written in Fortran, C, C++, and Python (via petsc4py; see :any:`sec-getting-started` ). PETSc
provides many of the mechanisms needed within parallel application
codes, such as parallel matrix and vector assembly routines. The library
is organized hierarchically, enabling users to employ the level of
abstraction that is most appropriate for a particular problem. By using
techniques of object-oriented programming, PETSc provides enormous
flexibility for users.

PETSc is a sophisticated set of software tools; as such, for some users
it initially has a much steeper learning curve than packages such as MATLAB or a simple subroutine
library. In particular, for individuals without some computer science
background, experience programming in C, C++, python, or Fortran and
experience using a debugger such as ``gdb`` or ``lldb``, it may require a
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
   functionality before starting to code directly with PETSc.

-  PETSc should *not* be used to attempt to provide a “parallel linear
   solver” in an otherwise sequential code. Certainly all parts of a
   previously sequential code need not be parallelized but the matrix
   generation portion must be parallelized to expect any kind of
   reasonable performance. Do not expect to generate your matrix
   sequentially and then “use PETSc” to solve the linear system in
   parallel.

Since PETSc is under continued development, small changes in usage and
calling sequences of routines will occur. PETSc has been supported for twenty-five years; see
https://www.mcs.anl.gov/petsc/miscellaneous/mailing-lists.html for
information on contacting support.

.. include :: ../common/editing-online.rst

Manual pages and example usage :
https://www.mcs.anl.gov/petsc/documentation/

Installing PETSc :
https://www.mcs.anl.gov/petsc/documentation/installation.html

Tutorials :
https://www.mcs.anl.gov/petsc/documentation/tutorials/index.html

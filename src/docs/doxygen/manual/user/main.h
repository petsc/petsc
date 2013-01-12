


/**
*   
*  \page usermanual User Manual

This manual describes the use of %PETSc for the numerical solution
of partial differential equations and related problems
on high-performance computers.  The
Portable, Extensible Toolkit for Scientific Computation (%PETSc) is a
suite of data structures and routines that provide the building
blocks for the implementation of large-scale application codes on parallel
(and serial) computers.  PETSc uses the MPI standard for all
message-passing communication.

%PETSc includes an expanding suite of parallel linear, nonlinear
equation solvers and time integrators that may be
used in application codes written in Fortran, C, C++, Python, and MATLAB (sequential).  %PETSc
provides many of the mechanisms needed within parallel application
codes, such as parallel matrix and vector assembly routines. The library is
organized hierarchically, enabling users to employ the level of
abstraction that is most appropriate for a particular problem. By
using techniques of object-oriented programming, PETSc provides
enormous flexibility for users.

%PETSc is a sophisticated set of software tools; as such, for some
users it initially has a much steeper learning curve than a simple
subroutine library. In particular, for individuals without some
computer science background, experience programming in C, C++ or Fortran and experience using a debugger such as \trl{gdb} or \trl{dbx}, it
may require a significant amount of time to take full advantage of the
features that enable efficient software use.  However, the power of
the PETSc design and the algorithms it incorporates may make the efficient
implementation of many application codes simpler than "rolling
them" yourself.
  -  For many tasks a package such as MATLAB is often the best tool; PETSc is not
intended for the classes of problems for which effective MATLAB code
can be written. PETSc also has a MATLAB interface, so portions of your code can be written in MATLAB to "try out" the PETSc solvers.
The resulting code will not be scalable however because currently MATLAB is inherently not scalable.
  - %PETSc should not be used to attempt to provide
a "parallel linear solver" in an otherwise sequential code.
Certainly all parts of a previously sequential code need not be parallelized but the
matrix generation portion must be parallelized to expect any kind of reasonable performance.
Do not expect to generate your matrix sequentially and then "use PETSc" to solve
the linear system in parallel.

Since %PETSc is under continued development, small changes in usage and
calling sequences of routines will occur.  %PETSc is supported; see the
web site http://www.mcs.anl.gov/petsc for information on
contacting support.

At http://www.mcs.anl.gov/petsc/publications
a list of publications and web sites may be found that feature work involving %PETSc.


We welcome any reports of corrections for this document.

\section Getting Information on PETSc

**Online:**
  - Manual pages and  example usage: `docs/index.html` or http://www.mcs.anl.gov/petsc/documentation
  - Installing PETSc: http://www.mcs.anl.gov/petsc/documentation/installation.html


**In this manual:**
  - \ref manual-user-page-getting-started "Basic introduction"
  - Assembling \ref manual-user-page-vectors " vectors " and \ref manual-user-page-matrices " matrices"
  - \ref manual-user-page-ksp "Linear solvers"
  - \ref manual-user-page-snes "Nonlinear solvers"
  - \ref manual-user-page-ts "Timestepping (ODE) solvers"

This manual consists of multiple parts:
   - \subpage manual-user-page-introduction
   - \subpage manual-user-page-programming-with-petsc
   - \subpage manual-user-page-additional-information
   - \subpage manual-user-page-acknowledgments

*/


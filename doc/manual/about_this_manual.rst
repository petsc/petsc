About This Manual
-----------------

This manual describes the use of the Portable, Extensible Toolkit for Scientific Computation
(PETSc) and the Toolkit for Advanced Optimization (TAO) for the numerical solution of
partial differential equations (PDEs) and related problems on high-performance
computers. PETSc/TAO is a suite of data structures and routines that provide the
building blocks for implementing large-scale application codes
on parallel (and serial) computers. PETSc uses the MPI standard for all
distributed memory communication.

PETSc/TAO includes a large suite of parallel **linear solvers**, **nonlinear
solvers**, **time integrators**, and **optimizers** that may be used in application codes
written in Fortran, C, C++, and Python (via petsc4py; see :any:`sec-getting-started` ). The library
is organized hierarchically, enabling users to employ the abstraction level most appropriate for a particular problem. By using
techniques of object-oriented programming, PETSc provides enormous
flexibility for users.

PETSc is a sophisticated set of software tools; 
it initially has a steeper learning curve than packages such as MATLAB or a simple subroutine
library. In particular, for individuals without some experience programming in C, C++, Python, or Fortran and
experience using a debugger such as ``gdb`` or ``lldb``, it may require a
significant amount of time to take full advantage of the features that
enable efficient software use. However, the power of the PETSc design
and the algorithms it incorporates makes the efficient implementation
of many application codes simpler than “rolling them” yourself.

-  For many tasks, a package such as MATLAB is often the best tool; PETSc
   is not intended for the classes of problems for which effective
   MATLAB code can be written.

- Several packages (listed on https://petsc.org/),
   built on PETSc, may satisfy your needs without requiring
   directly using PETSc. We recommend reviewing these packages'
   functionality before starting to code directly with PETSc.

-  PETSc can be used to provide a “MPI parallel linear
   solver” in an otherwise sequential or OpenMP parallel code.
   This approach can provide modest improvements in the application time
   by utilizing modest numbers of MPI processes. See ``PCMPI`` for details on how to
   utilize the PETSc MPI linear solver server.

Since PETSc is under continued development, small changes in usage and
calling sequences of routines will occur. PETSc has been supported for twenty-five years; see
:doc:`mailing list information on our website </community/mailing>` for
information on contacting support.

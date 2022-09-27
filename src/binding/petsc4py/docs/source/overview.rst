Overview
========

PETSc_ is a suite of data structures and routines for the
scalable (parallel) solution of scientific applications modeled by
partial differential equations. It employs the MPI_ standard for all
message-passing communication.

PETSc is intended for use in large-scale application projects
[petsc-efficient]_, and several ongoing computational science projects
are built around the PETSc libraries. With strict attention to
component interoperability, PETSc facilitates the integration of
independently developed application modules, which often most
naturally employ different coding styles and data structures.

PETSc is easy to use for beginners [petsc-user-ref]_. Moreover, its
careful design allows advanced users to have detailed control over the
solution process. PETSc includes an expanding suite of parallel linear
and nonlinear equation solvers that are easily used in application
codes written in C, C++, and Fortran. PETSc provides many of the
mechanisms needed within parallel application codes, such as simple
parallel matrix and vector assembly routines that allow the overlap of
communication and computation.

.. [petsc-user-ref] S. Balay, S. Abhyankar, M. Adams, S. Benson, J. Brown,
  P. Brune, K. Buschelman, E. Constantinescu, L. Dalcin, A. Dener,
  V. Eijkhout, J. Faibussowitsch, W. Gropp, V. Hapla, T. Isaac, P. Jolivet,
  D. Karpeyev, D. Kaushik, M. Knepley, F. Kong, S. Kruger,
  D. May, L. Curfman McInnes, R. Mills, L. Mitchell, T. Munson,
  J. Roman, K. Rupp, P. Sanan, J Sarich, B. Smith,
  S. Zampini, H. Zhang, and H. Zhang, J. Zhang,
  *PETSc/TAO Users Manual*, ANL-21/39 - Revision 3.18, 2022.
  https://petsc.org/release/docs/manual/manual.pdf

.. [petsc-efficient] Satish Balay, Victor Eijkhout, William D. Gropp,
   Lois Curfman McInnes and Barry F. Smith. Efficient Management of
   Parallelism in Object Oriented Numerical Software Libraries. Modern
   Software Tools in Scientific Computing. E. Arge, A. M. Bruaset and
   H. P. Langtangen, editors. 163--202. Birkhauser Press. 1997.

.. include:: links.txt


Components
----------

PETSc is designed with an object-oriented style. Almost all
user-visible types are abstract interfaces with implementations that
may be chosen at runtime. Those objects are managed through handles to
opaque data structures which are created, accessed and destroyed by
calling appropriate library routines.

PETSc consists of a variety of components. Each component manipulates
a particular family of objects and the operations one would like to
perform on these objects. These components provide the functionality
required for many parallel solutions of PDEs.

:Vec:  Provides the vector operations required for setting up and
       solving large-scale linear and nonlinear problems. Includes
       easy-to-use parallel scatter and gather operations, as well as
       special-purpose code for handling ghost points for regular data
       structures.

:Mat:  A large suite of data structures and code for the manipulation
       of parallel sparse matrices. Includes four different parallel
       matrix data structures, each appropriate for a different class
       of problems.

:PC:   A collection of sequential and parallel preconditioners,
       including (sequential) ILU(k), LU, and (both sequential and
       parallel) block Jacobi, overlapping additive Schwarz methods
       and (through BlockSolve95) ILU(0) and ICC(0).

:KSP:  Parallel implementations of many popular Krylov subspace
       iterative methods, including GMRES, CG, CGS, Bi-CG-Stab, two
       variants of TFQMR, CR, and LSQR. All are coded so that they are
       immediately usable with any preconditioners and any matrix data
       structures, including matrix-free methods.

:SNES: Data-structure-neutral implementations of Newton-like methods
       for nonlinear systems. Includes both line search and trust
       region techniques with a single interface. Employs by default
       the above data structures and linear solvers. Users can set
       custom monitoring routines, convergence criteria, etc.

:TS:   Code for the time evolution of solutions of PDEs. In addition,
       provides pseudo-transient continuation techniques for computing
       steady-state solutions.

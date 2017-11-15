/**
*   \page dev-petsc-objects PetscObjects

\section dev-petsc-objects-elementary Elementary Objects: IS, Vec, Mat

\section dev-petsc-objects-solver Solver Objects: PC, KSP, SNES, TS

\subsection dev-petsc-objects-preconditioners Preconditioners: PC

The base %PETSc PC object is defined in `include/petsc-private/pcimpl.h`.
A carefully commented implementation of a PC object can be found in
`src/ksp/pc/impls/jacobi/jacobi.c`.


\subsection dev-petsc-objects-krylov Krylov Solvers: KSP
The base %PETSc KSP object is defined in `include/petsc-private/kspimpl.h`.
A carefully commented implementation of a KSP object can be found in
`src/ksp/ksp/impls/cg/cg.c`.

\subsection dev-petsc-objects-registering Registering New Methods


See `src/ksp/examples/tutorials/ex12.c` for an example of registering a new
preconditioning (PC) method.

*/

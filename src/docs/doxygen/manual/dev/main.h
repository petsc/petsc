
/**
*  \page devmanual Developer Manual
PETSc is a set of extensible software libraries for scientific computation.
PETSc is designed using a object-oriented
architecture. This means that libraries consist of *objects* that
have certain, defined functionality. This document defines how these
objects are implemented.

This manual discusses the PETSc library design and describes
how to develop new library codes that are compatible with other PETSc components
including PETSc 3.0.
The idea is not to develop one massive library that everyone shoves code
into; rather, to develop an architecture that allows many people
to (as painlessly as possible) contribute (and maintain) their own libraries,
in a distributed fashion.

The text assumes
that you are familiar with PETSc, have a copy of the PETSc users
manual, and have access to PETSc source code and documentation
(available via http://www.mcs.anl.gov/petsc )

Please direct all comments and questions regarding PETSc design and
development to petsc-dev@mcs.anl.gov.  Note that 
**all bug reports and questions regarding the use of PETSc** should continue
to be directed to petsc-maint@mcs.anl.gov.

   - \subpage dev-petsc-kernel
   - \subpage dev-basic-object-design
   - \subpage dev-minimal-class-standards
   - \subpage dev-petsc-objects
   - \subpage dev-style-guide
   - \subpage dev-matrix-classes
*
*/

===================
PETSc in a nutshell
===================

PETSc/TAO is a tool for writing, analyzing, and optimizing properties of large-scale numerical simulations.

Algebraic objects
=================

- ``Vec`` - containers for simulation solutions, right hand sides of linear systems, etc.

- ``Mat`` - contain Jacobians and operators that define linear systems

- ``IS`` indices - used to access portions of vectors and matrix, for example {1,2,4} or 1:10

Solvers
=======

- ``PC`` preconditioners - approximate solvers to algebra systems without a history of previous iterations

- ``KSP`` Krylov Subspace methods - approximate solvers with a history of previous iterations

- ``SNES`` nolinear equation solver

- ``TS`` time integrators (ODE/PDE), explicit, implicit, local and global error estimators

- ``TSAdjoint`` derivatives/sensitivities of functions of ODE/PDE integration solutions

- ``TAO`` - optimization, with equality and inequality constraints, first and second order (Newton) methods

Connectors of continuum models, meshes, and discretizations to solvers and algebraic objects
============================================================================================

- ``DMDA`` - for simulations computed on simple structured grids

- ``DMSTAG`` - for simulations computed on staggered grids

- ``DMPLEX``  - for simulation computed on unstructured meshes

- ``DMNETWORK`` - for simulations on networks or graphs, for example the power grid, river networks, the nervous system

- ``DMP4EST`` - for simulations on collections of quad or octree meshes


Utilities
=========

- ``PetscOptions`` - control of discretization and solution process

- ``PetscViewer`` - visualizing algebraic objects, solvers, connectors

- Monitor - monitoring of solution progress

- ``Profiling`` - profiling of the performance of the simulation solution process


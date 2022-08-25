===================
PETSc in a nutshell
===================

PETSc/TAO is a tool for writing, analyzing, and optimizing properties of large-scale numerical simulations.

.. image:: /images/docs/manual/library_structure.svg
   :alt: PETSc Structure Diagram
   :align: center

Algebraic objects
=================

* ``Vec`` - containers for simulation solutions, right hand sides of linear systems, etc. (:any:`chapter_vectors`)

* ``Mat`` - contain Jacobians and operators that define linear systems (:any:`chapter_matrices`)

  * Several sparse and dense matrix storage formats (see ``MatType``):

  * Limited memory variable metric matrix representations

  * Block

  * Nested

  * :ref:`Easy, efficient matrix assembly and interface <sec_matcreate>`

* ``IS`` indices - used to access portions of vectors and matrix, for example {1,2,4} or 1:10

Solvers
=======

* ``PC`` preconditioners - approximate solvers to algebra systems without a history of previous iterations

* ``KSP`` Krylov subspace methods - approximate solvers with a history of previous iterations (:any:`chapter_ksp`)

* ``SNES`` nonlinear equation solvers (:any:`chapter_snes`)

* ``TS`` time integrators (ODE/PDE), explicit, implicit, local and global error estimators (:any:`chapter_ts`)

  * Local and global error estimators

  * ``TSAdjoint`` derivatives/sensitivities of functions of ODE/PDE integration solutions (:any:`section_sa`)

* ``TAO`` - optimization, with equality and inequality constraints, first and second order (Newton) methods (:any:`chapter_tao`)

.. seealso::

   For full feature list and prerequisites see:

   - :ref:`Linear solver table <doc_linsolve>`
   - :ref:`Nonlinear solver table <doc_nonlinsolve>`
   - :ref:`Tao solver table <doc_taosolve>`

DM: Interfacing Between Solvers and Models/Discretizations
==========================================================

* ``DMDA`` - for simulations computed on simple structured grids

* ``DMSTAG`` - for simulations computed on staggered grids (:any:`chapter_stag`)

* ``DMPLEX``  - for simulation computed on unstructured meshes (:any:`chapter_unstructured`)

* ``DMNETWORK`` - for simulations on networks or graphs, for example the power grid, river networks, the nervous system (:any:`chapter_network`)

* ``DMP4EST`` - for simulations on collections of quad or octree meshes

* ``DMSWARM`` - for simulations on particles


Utilities
=========

* ``PetscOptions`` - control of discretization and solution process

* ``PetscViewer`` - visualizing algebraic objects, solvers, connectors

* Monitor - monitoring of solution progress

* ``Profiling`` - profiling of the performance of the simulation solution process

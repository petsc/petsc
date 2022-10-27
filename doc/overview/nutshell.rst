===================
PETSc in a nutshell
===================

PETSc/TAO is a tool for writing, analyzing, and optimizing properties of large-scale numerical simulations.

.. image:: /images/docs/manual/library_structure.svg
   :alt: PETSc Structure Diagram
   :align: center

Algebraic objects
=================

* :any:`Vectors <chapter_vectors>` - containers for simulation solutions, right hand sides of linear systems, etc (``Vec``).

* :any:`Matrices <chapter_matrices>`  - contain Jacobians and operators that define linear systems (``Mat``).

  * Multiple sparse and dense matrix storage formats,

  * limited memory variable metric representations,

  * block and nested representations (see ``MatType``).

  * :ref:`Easy, efficient matrix assembly and interface <sec_matcreate>`

* Indices - used to access portions of vectors and matrix, for example {1,2,4} or 1:10 (``IS``).

Solvers
=======

* :any:`Linear solvers<chapter_ksp>` based on preconditioners (``PC``) and Krylov subspace methods (``KSP``).

* :any:`Nonlinear solvers <chapter_snes>` (``SNES``).

* :any:`Time integrators <chapter_ts>`, (ODE/PDE), explicit, implicit, IMEX, (``TS``)

  * Local and global error estimators

  * :any:`section_sa`.

* :any:`Optimization <chapter_tao>` with equality and inequality constraints, first and second order (Newton) methods (``Tao``).

.. seealso::

   For full feature list and prerequisites see:

   - :ref:`Vector table <doc_vector>`
   - :ref:`Matrix table <doc_matrix>`
   - :ref:`Linear solver table <doc_linsolve>`
   - :ref:`Nonlinear solver table <doc_nonlinsolve>`
   - :ref:`Tao solver table <doc_taosolve>`

DM: Interfacing Solvers to Models/Discretizations
==========================================================

* ``DMDA`` - for simulations computed on simple structured grids

* :any:`chapter_stag` - for simulations computed on staggered grids, (``DMSTAG``)

* :any:`chapter_unstructured` - for simulation computed on unstructured meshes, (``DMPLEX``)

* :any:`chapter_network` - for simulations on networks or graphs, for example the power grid, river networks, the nervous system, (``DMNETWORK``)

* ``DMP4EST`` - for simulations on collections of quad or octree meshes

* ``DMSWARM`` - for simulations on particles


Utilities for the Simulation/Solver Process
===========================================

Runtime

* control of the simulation, :any:`sec_options`

* visualization of the solvers and simulation, :any:`sec_viewers`,

* monitoring of solution progress,

*  :any:`ch_profiling` of the performance,

* robust :any:`sec_errors`.

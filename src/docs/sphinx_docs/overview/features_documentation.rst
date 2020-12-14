.. _doc_features:

*******************************
Core Features and Functionality
*******************************

PETSc is a general parallel linear and non-linear solver framework, which provides these
general classes of functionality:

.. contents:: Table Of Contents
   :local:
   :backlinks: entry
   :depth: 1


Supported Systems
=================

- :ref:`HPC <doc_config_hpc>`
- :ref:`Linux <doc_config_faq>`
- :ref:`MacOS <doc_config_faq>`
- :ref:`Windows <doc_windows>`

General Features
================

- :ref:`Parallel vectors <chapter_vectors>`
- :ref:`Vector code for communicating ghost points <sec_scatter>`
- :ref:`Parallel matrices <chapter_matrices>`
- Several sparse and dense matrix storage formats (see ``MatType``):

  - AIJ/CSR (Yale sparse matrix format)
  - LMVM (Limited Memory Variable Metric)
  - Block
  - Nested
  - Dense

- :ref:`Easy, efficient matrix assembly and interface <sec_matcreate>`
- :ref:`Powerful object introspection tools <sec_viewers>`
- Support for a variety of IO options (see ``PetscViewerType``)
- :ref:`Comprehensive performance testing <ch_performance>`

Solver Features
===============

- :ref:`Parallel Krylov subspace methods <chapter_ksp>`
- :ref:`Parallel nonlinear solvers <chapter_snes>`
- Scalable parallel :ref:`linear <sec_ksppc>` and :ref:`nonlinear <sec_snespc>`
  preconditioners
- :ref:`Parallel timestepping (ODE) solvers <chapter_ts>`
- Local and global error estimators
- :ref:`Forward and adjoint sensitivity capabilities <chapter_sa>`
- Robust optimization through ``Tao``

.. seealso::

   For full feature list and prerequisites see:

   - :ref:`Linear solver table <doc_linsolve>`
   - :ref:`Nonlinear solver table <doc_nonlinsolve>`
   - :ref:`Tao solver table <doc_taosolve>`

Accelerator/GPU Features
========================

- :ref:`Matrix/Vector CUDA support <doc_config_accel_cuda>`
- :ref:`Kokkos support <doc_config_accel_kokkos>`
- :ref:`Matrix/Vector OpenCL/ViennaCL support <doc_config_accel_opencl>`
- :ref:`Matrix/Vector HIP support <doc_gpu_roadmap>`

.. note::

   PETSc GPU support is under heavy development! See GPU support :ref:`roadmap
   <doc_gpu_roadmap>` for more information on current support.

Support Features
================

- Complete documentation
- :ref:`Comprehensive profiling of floating point and memory usage <ch_profiling>`
- Consistent user interface
- :ref:`Intensive error checking <sec_errors>`
- Over one thousand examples
- :ref:`PETSc is supported and will be actively enhanced for many years
  <doc_faq_maintenance_strats>`

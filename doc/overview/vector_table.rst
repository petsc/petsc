.. _doc_vector:

==========================================
Summary of Vector Types Available In PETSc
==========================================

.. list-table::
   :widths: auto
   :align: center
   :header-rows: 1

   * - Format
     - Vector Types
     - External Packages
     - Details
   * - Dense array
     - ``VECSTANDARD``
     - BLAS
     -
   * -
     - ``VECCUDA``
     - NVIDIA's cuBLAS
     - NVIDIA GPUs
   * -
     - ``VECHIP``
     - AMD's RocBLAS
     - AMD GPUs
   * -
     - ``VECKOKKOS``
     - Kokkos
     - GPUs, CPUs, OpenMP
   * - Nested
     - ``VECNEST``
     -
     - Provides efficient access to inner vectors

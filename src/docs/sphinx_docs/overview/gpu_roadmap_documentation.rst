.. _doc_gpu_roadmap:

*******************
GPU Support Roadmap
*******************

PETSc algebraic solvers run on GPU systems from NVIDIA using CUDA, and AMD and Intel using
OpenCL/ViennaCL and HIP. Effective GPU implementations of low-level linear algebra
operations provide a highly performant alternative solution strategy for users, and is
therefore a high priority for PETSc developers.

See FAQ :ref:`topic <doc_faq_gpuhowto>` which shows how to enable GPU backends.

.. note::

   PETSc uses a single source programming model where solver back-ends are selected as
   **runtime** options and configuration options with no changes to the API.

   Users should (ideally) never have to change their source code to take advantage of new
   backend implementations.

PETSc code will include full implementations of vector and matrix operations (as well as
other select operations) using each of:

.. list-table::
   :widths: auto
   :header-rows: 1

   * - Language/Programming Model
     - Supporting Package
     - ``Vec`` Status
     - ``Mat`` Status
   * - CUDA
     - cuBLAS/cuSparse
     - SUPPORTED
     - SUPPORTED
   * - HIP
     - Rocm
     - SUPPORTED
     - IN DEVELOPMENT
   * - SYCL
     - MKL
     - NOT YET SUPPORTED
     - NOT YET SUPPORTED
   * - OpenCL
     - ViennaCL
     - SUPPORTED
     - SUPPORTED
   * - Kokkos
     - ---
     - SUPPORTED
     - IN DEVELOPMENT

---------------------------------

Basic linear algebra GPU implementations enable many solvers, including ``GAMG`` and
``BDDC``, to run entirely on the GPU. PETSc is currently adding GPU support for residual
and Jacobian creation and for matrix assembly extensions to ``MATAIJCUSPARSE`` and
``MATAIJKOKKOS``. **This is work in progress**.

.. admonition:: Important
   :class: yellow

   We could use your help in further developing PETSc for GPUs; see PETSc Developers
   :ref:`documentation <ind_developers>`. The label ``GPU`` is used on our `Gitlab
   <https://gitlab.com/petsc/petsc>`__ repository for all activity involving GPUs.

   **You must use petsc master (git branch) for GPUs, do not install the current release.**

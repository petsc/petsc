(doc_gpu_roadmap)=

# GPU Support Roadmap

PETSc provides GPU vector and matrix implementations using CUDA for NVIDIA GPUs,
HIP for AMD GPUs, and Kokkos with CUDA, HIP, or SYCL for NVIDIA, AMD, or Intel GPUs,
respectively. OpenCL support through ViennaCL is also available as a
{ref}`legacy backend <doc_config_accel_opencl>`.

See FAQ {ref}`topic <doc_faq_gpuhowto>` which shows how to enable GPU backends.

:::{note}
PETSc uses a single source programming model where solver back-ends are selected as
**runtime** options and configuration options with no changes to the API.

Users should (ideally) never have to change their source code to take advantage of new
backend implementations.
:::

The following backends provide vector and matrix operations. Coverage varies
by operation and matrix type.

```{eval-rst}
.. list-table::
   :widths: auto
   :header-rows: 1

   * - Programming Model
     - Supporting Package
     - ``Vec`` Status
     - ``Mat`` Status
     - Supported GPU types
   * - CUDA
     - cuBLAS/cuSPARSE
     - SUPPORTED
     - SUPPORTED
     - NVIDIA GPUs
   * - HIP
     - ROCm
     - SUPPORTED
     - SUPPORTED
     - AMD GPUs
   * - SYCL
     - Kokkos and Kokkos Kernels
     - SUPPORTED via Kokkos
     - SUPPORTED via Kokkos
     - Intel GPUs
   * - OpenCL
     - ViennaCL
     - LEGACY SUPPORT
     - LEGACY SUPPORT
     - NVIDIA, AMD, Intel GPUs
   * - Kokkos
     - Kokkos and Kokkos Kernels
     - SUPPORTED
     - SUPPORTED
     - NVIDIA, AMD, Intel GPUs
```

______________________________________________________________________

Most Krylov methods use the selected vector and matrix implementations for their
linear algebra. GPU execution of a preconditioner depends on its components,
including smoothers and coarse solvers. Selecting GPU vectors and matrices does
not make application residual or Jacobian callbacks execute on the GPU.

`MatSetPreallocationCOO()` and `MatSetValuesCOO()` support GPU matrix assembly with
`MATAIJCUSPARSE`, `MATAIJHIPSPARSE`, and `MATAIJKOKKOS`. Applications can compute
the values on the device and pass them to `MatSetValuesCOO()`.
Use `-log_view` to inspect GPU work and host-device transfers, and see
{any}`ch_device` for stream ordering and current limitations.

:::{important}
:class: yellow

We could use your help in further developing PETSc for GPUs; see PETSc Developers
{ref}`documentation <ind_developers>`. The label `GPU` is used on our [GitLab](https://gitlab.com/petsc/petsc) repository for all activity involving GPUs.
:::

#include <../src/ml/da/impls/ensemble/letkf/letkf.h>
#include <Kokkos_Core.hpp>
#include <KokkosBlas.hpp>
#include <climits>

#if defined(KOKKOS_ENABLE_CUDA)
  #include <cusolverDn.h>
  #include <cuda_runtime.h>
  #include <petscdevice_cuda.h>
#elif defined(KOKKOS_ENABLE_HIP)
  #include <rocsolver/rocsolver.h>
  #include <hip/hip_runtime.h>
  #include <petscdevice_hip.h>
#elif defined(KOKKOS_ENABLE_SYCL)
  #include <oneapi/mkl.hpp>
  #include <sycl/sycl.hpp>
#endif

/* Shared device-View aliases used throughout the BatchedEigenSolve* dispatch chain and
   by the per-mirror device locals in PetscDALETKFLocalAnalysis_Kokkos. */
using LETKFExecSpace = Kokkos::DefaultExecutionSpace;
using LETKFView3D    = Kokkos::View<PetscScalar ***, Kokkos::LayoutLeft, LETKFExecSpace>;
using LETKFView2D    = Kokkos::View<PetscScalar **, Kokkos::LayoutLeft, LETKFExecSpace>;
using LETKFView1D    = Kokkos::View<PetscScalar *, Kokkos::LayoutLeft, LETKFExecSpace>;

/* Floor used when dividing by Lambda_i from the per-vertex eigendecomposition of
   T = (1/rho)*I + S^T*S. T is SPD by construction so eigenvalues are positive in exact arithmetic,
   but device eigensolvers (cusolver syevj, rocsolver syevd, oneMKL syevd) can produce tiny
   rounding-magnitude eigenvalues on near-degenerate spectra; without a floor, dividing by them
   produces inf/nan in temp2 and inv_sqrt_lambda_i and silently corrupts the analysis in release
   builds (the DEBUG CheckLambda block only flags eigenvalues below -1e-8, which lets near-zero
   positives pass through). Scaled by PETSC_MACHINE_EPSILON so the floor follows precision; the
   absolute floor matches the original 1.0e-14 in double precision and adapts down/up for
   single/quad. */
static constexpr PetscReal LETKF_EIGEN_EPS = (PetscReal)100.0 * PETSC_MACHINE_EPSILON;

/* ========================================================================== */
/*                    Batched Eigendecomposition for LETKF                    */
/* ========================================================================== */

/* Structure to hold reusable workspace for eigensolvers.
   Lifecycle is manual: allocated in PetscDALETKFLocalAnalysis_Kokkos/GlobalAnalysis_Kokkos when
   first needed and freed in PetscDALETKFDestroyLocalization_Kokkos. We do not provide a
   destructor because the cleanup uses CUDA/HIP/SYCL APIs that need PETSc error checking
   (PetscCallCUDA/HIP, sycl::free with a queue), which cannot be expressed inside a C++
   destructor body that is supposed to be noexcept. */
struct EigenWorkspace {
  /* Tracking for reuse */
  PetscInt max_chunk_size;
  PetscInt m;
  PetscInt max_nnz;

  /* Persistent Kokkos Views */
  using exec_space = Kokkos::DefaultExecutionSpace;
  using view_3d    = Kokkos::View<PetscScalar ***, Kokkos::LayoutLeft, exec_space>;
  using view_2d    = Kokkos::View<PetscScalar **, Kokkos::LayoutLeft, exec_space>;

  view_3d S_batch;
  view_3d T_batch;
  view_3d V_batch;
  view_2d Lambda_batch;
  view_3d T_sqrt_batch;
  view_2d w_batch;
  view_2d delta_batch;
  view_2d y_batch;
  view_2d y_mean_batch;
  view_2d r_inv_sqrt_batch;
  view_2d temp1_batch;
  view_2d temp2_batch;
  view_2d inv_sqrt_lambda_batch;

  /* Host workspace */
  PetscScalar *all_v;
  PetscReal   *all_lambda;
  PetscScalar *all_work;
#if defined(PETSC_USE_COMPLEX)
  PetscReal *all_rwork;
#endif
  PetscBLASInt lwork;
  PetscBLASInt n_blas;

  /* Device workspace */
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP) || defined(KOKKOS_ENABLE_SYCL)
  #if defined(KOKKOS_ENABLE_CUDA)
  syevjInfo_t  syevj_params;
  PetscScalar *d_work;
  int         *d_info;
  PetscScalar *d_A_contig;
  PetscScalar *d_W_contig;
  int          lwork_device;
  #elif defined(KOKKOS_ENABLE_HIP)
  PetscScalar *d_work;
  int         *d_info;
  PetscScalar *d_A_contig;
  PetscScalar *d_W_contig;
  int          lwork_device;
  #elif defined(KOKKOS_ENABLE_SYCL)
  PetscScalar *d_work;
  int         *d_info;
  PetscScalar *d_A_contig;
  PetscScalar *d_W_contig;
  int          lwork_device;
  #endif
#endif

  EigenWorkspace() : max_chunk_size(0), m(0), max_nnz(0), all_v(nullptr), all_lambda(nullptr), all_work(nullptr)
  {
#if defined(PETSC_USE_COMPLEX)
    all_rwork = nullptr;
#endif
#if defined(KOKKOS_ENABLE_CUDA)
    d_work       = nullptr;
    d_info       = nullptr;
    d_A_contig   = nullptr;
    d_W_contig   = nullptr;
    syevj_params = nullptr;
#elif defined(KOKKOS_ENABLE_HIP) || defined(KOKKOS_ENABLE_SYCL)
    d_work     = nullptr;
    d_info     = nullptr;
    d_A_contig = nullptr;
    d_W_contig = nullptr;
#endif
  }
};

/*
  BatchedEigenSolve_Host - Compute eigendecomposition for a batch of symmetric matrices (CPU version)

  Input Parameters:
+ T_batch      - batch of symmetric matrices (n_batch x n_size x n_size)
. n_batch      - number of matrices in the batch
- n_size       - size of each matrix (m x m)
- work         - reusable workspace structure

  Output Parameters:
+ Lambda_batch - eigenvalues for each matrix (n_batch x n_size)
- V_batch      - eigenvectors for each matrix (n_batch x n_size x n_size)

  Notes:
  Uses LAPACK's syev routine to compute eigendecomposition sequentially on host.
*/
#if !defined(KOKKOS_ENABLE_CUDA) && !defined(KOKKOS_ENABLE_HIP) && !defined(KOKKOS_ENABLE_SYCL)
  #include <petscblaslapack.h>
static PetscErrorCode BatchedEigenSolve_Host(LETKFView3D T_batch, LETKFView2D Lambda_batch, LETKFView3D V_batch, PetscInt n_batch, PetscInt n_size, EigenWorkspace *work)
{
  PetscFunctionBegin;
  /* In the host-only build path the batch views already live in HostSpace, so LAPACK can
     read T_batch and write Lambda_batch/V_batch directly. The mirror+deep_copy round-trip
     used in the device path would be a no-op here, and some Kokkos+Serial configurations do
     not expose ::HostMirror on a View parameterized with an exec-space tag. */

  /* Use pre-allocated workspace */
  PetscScalar *all_v      = work->all_v;
  PetscReal   *all_lambda = work->all_lambda;
  PetscScalar *all_work   = work->all_work;
  PetscBLASInt lwork      = work->lwork;
  PetscBLASInt n_blas     = work->n_blas;
  #if defined(PETSC_USE_COMPLEX)
  PetscReal *all_rwork = work->all_rwork;
  #endif

  /* Process each matrix in parallel on host using LAPACK */
  Kokkos::parallel_for(
    "BatchedEigenSolve_Host", Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0, n_batch), KOKKOS_LAMBDA(const int i) {
      PetscBLASInt n   = n_blas;
      PetscBLASInt lda = n;
      PetscBLASInt info;
      PetscBLASInt lw = lwork;

      /* Pointers for this matrix */
      PetscScalar *v_ptr      = all_v + i * n_size * n_size;
      PetscReal   *lambda_ptr = all_lambda + i * n_size;
      PetscScalar *work_ptr   = all_work + i * lwork;
  #if defined(PETSC_USE_COMPLEX)
      PetscReal *rwork_ptr = all_rwork + i * (3 * n_size - 2);
  #endif

      /* Copy T_batch(i, :, :) to v_ptr (column-major) */
      for (PetscInt j = 0; j < n_size; j++) {
        for (PetscInt k = 0; k < n_size; k++) v_ptr[k + j * n_size] = T_batch(i, k, j);
      }

    /* Compute eigendecomposition: T = V * Lambda * V^T */
  #if defined(PETSC_USE_COMPLEX)
      LAPACKsyev_("V", "U", &n, v_ptr, &lda, lambda_ptr, work_ptr, &lw, rwork_ptr, &info);
  #else
      LAPACKsyev_("V", "U", &n, v_ptr, &lda, lambda_ptr, work_ptr, &lw, &info);
  #endif

      /* Kokkos::parallel_for() cannot return error codes; abort the parallel region instead. */
      if (info != 0) Kokkos::abort("LAPACK eigendecomposition failed in parallel region");

      /* Write results directly back to the batch views (already in HostSpace) */
      for (PetscInt j = 0; j < n_size; j++) {
        Lambda_batch(i, j) = (PetscScalar)lambda_ptr[j];
        for (PetscInt k = 0; k < n_size; k++) V_batch(i, k, j) = v_ptr[k + j * n_size];
      }
    });
  PetscFunctionReturn(PETSC_SUCCESS);
}
#endif

/*
  BatchedEigenSolve_Device - Compute eigendecomposition for a batch of symmetric matrices (Device version)

  Input Parameters:
+ T_batch      - batch of symmetric matrices (n_batch x n_size x n_size)
. n_batch      - number of matrices in the batch
- n_size       - size of each matrix (m x m)
- device_handle - device-specific solver handle (cusolverDnHandle_t, rocblas_handle, or sycl::queue*)
- work         - reusable workspace structure

  Output Parameters:
+ Lambda_batch - eigenvalues for each matrix (n_batch x n_size)
- V_batch      - eigenvectors for each matrix (n_batch x n_size x n_size)

  Notes:
  Uses vendor-specific batched symmetric eigensolvers:
  - CUDA: cuSOLVER's syevjBatched
  - HIP: rocSOLVER's rocsolver_dsyevj_batched
  - SYCL: oneMKL's syevd_batch
*/
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP) || defined(KOKKOS_ENABLE_SYCL)
  #if defined(KOKKOS_ENABLE_CUDA)
static PetscErrorCode BatchedEigenSolve_Device(LETKFView3D T_batch, LETKFView2D Lambda_batch, LETKFView3D V_batch, PetscInt n_batch, PetscInt n_size, cusolverDnHandle_t cusolverH, EigenWorkspace *work)
{
  PetscFunctionBegin;
    #if defined(PETSC_USE_COMPLEX)
  /* cuSOLVER's *syevjBatched is real-only (Ssyevj/Dsyevj); under complex the call would type-error.
     The dispatcher gates the Kokkos path off when PETSC_USE_COMPLEX is set, so this is unreachable
     in practice; SETERRQ here as defense-in-depth in case that gate ever changes. */
  SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Complex numbers not supported on CUDA backend for LETKF");
    #else
  cusolverStatus_t cusolver_status;
  syevjInfo_t      syevj_params = work->syevj_params;
  PetscScalar     *d_work       = work->d_work;
  int             *d_info       = work->d_info;
  PetscScalar     *d_A_contig   = work->d_A_contig;
  PetscScalar     *d_W_contig   = work->d_W_contig;
  int              lwork        = work->lwork_device;
  int             *h_info       = nullptr;
  /* Copy T_batch to contiguous layout for cuSOLVER */
  Kokkos::parallel_for(
    "ReorganizeForCuSOLVER", Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, n_batch), KOKKOS_LAMBDA(const int i) {
      for (int j = 0; j < n_size; j++) {
        for (int k = 0; k < n_size; k++) d_A_contig[i * n_size * n_size + k * n_size + j] = T_batch(i, j, k);
      }
    });
  Kokkos::fence();

      /* Solve batched eigendecomposition */
      #if defined(PETSC_USE_REAL_SINGLE)
  cusolver_status = cusolverDnSsyevjBatched(cusolverH, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_UPPER, n_size, d_A_contig, n_size, d_W_contig, d_work, lwork, d_info, syevj_params, n_batch);
      #else
  cusolver_status = cusolverDnDsyevjBatched(cusolverH, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_UPPER, n_size, d_A_contig, n_size, d_W_contig, d_work, lwork, d_info, syevj_params, n_batch);
      #endif
  PetscCheck(cusolver_status == CUSOLVER_STATUS_SUCCESS, PETSC_COMM_SELF, PETSC_ERR_LIB, "cusolverDn*syevjBatched failed");

  /* Check info */
  PetscCall(PetscMalloc1(n_batch, &h_info));
  PetscCallCUDA(cudaMemcpy(h_info, d_info, sizeof(int) * n_batch, cudaMemcpyDeviceToHost));
  for (PetscInt i = 0; i < n_batch; i++) PetscCheck(h_info[i] == 0, PETSC_COMM_SELF, PETSC_ERR_LIB, "cuSOLVER eigendecomposition failed for matrix %" PetscInt_FMT ": info=%d", i, h_info[i]);
  PetscCall(PetscFree(h_info));

  /* Copy results back from contiguous layout to V_batch */
  Kokkos::parallel_for(
    "CopyResultsBack", Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, n_batch), KOKKOS_LAMBDA(const int i) {
      for (int j = 0; j < n_size; j++) {
        for (int k = 0; k < n_size; k++) V_batch(i, j, k) = d_A_contig[i * n_size * n_size + k * n_size + j];
        /* CUDA-12.6 nvcc compiler hangs if this line is placed before the V_batch loop. */
        Lambda_batch(i, j) = d_W_contig[i * n_size + j];
      }
    });
  Kokkos::fence();
    #endif
  PetscFunctionReturn(PETSC_SUCCESS);
}
  #elif defined(KOKKOS_ENABLE_HIP)
static PetscErrorCode BatchedEigenSolve_Device(LETKFView3D T_batch, LETKFView2D Lambda_batch, LETKFView3D V_batch, PetscInt n_batch, PetscInt n_size, rocblas_handle rocblasH, EigenWorkspace *work)
{
  PetscFunctionBegin;
    #if defined(PETSC_USE_COMPLEX)
  /* Bail out before any kernel launch: the workspace setup leaves d_A_contig/d_W_contig/d_work/d_info
     as nullptr in complex mode (rocsolver_*syevd has no complex variant we wrap), so the
     ReorganizeForRocSOLVER parallel_for below would do a null device write before this error fired. */
  SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Complex numbers not supported on HIP backend for LETKF");
    #else
  PetscScalar *d_work     = work->d_work;
  int         *d_info     = work->d_info;
  PetscScalar *d_A_contig = work->d_A_contig;
  PetscScalar *d_W_contig = work->d_W_contig;
  int         *h_info     = nullptr;

  /* Copy T_batch to contiguous layout for rocSOLVER */
  Kokkos::parallel_for(
    "ReorganizeForRocSOLVER", Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, n_batch), KOKKOS_LAMBDA(const int i) {
      for (int j = 0; j < n_size; j++) {
        for (int k = 0; k < n_size; k++) d_A_contig[i * n_size * n_size + k * n_size + j] = T_batch(i, j, k);
      }
    });
  Kokkos::fence();

  /* rocSOLVER doesn't have a native batched syevj, so we loop over batch.
     Use rocsolver_*syevd which is more efficient than calling syev in a loop. */
  for (int i = 0; i < n_batch; i++) {
    PetscScalar   *A_ptr    = d_A_contig + i * n_size * n_size;
    PetscScalar   *W_ptr    = d_W_contig + i * n_size;
    int           *info_ptr = d_info + i;
    rocblas_status hip_status;

      #if defined(PETSC_USE_REAL_SINGLE)
    hip_status = rocsolver_ssyevd(rocblasH, rocblas_evect_original, rocblas_fill_upper, n_size, A_ptr, n_size, W_ptr, d_work, info_ptr);
      #else
    hip_status = rocsolver_dsyevd(rocblasH, rocblas_evect_original, rocblas_fill_upper, n_size, A_ptr, n_size, W_ptr, d_work, info_ptr);
      #endif
    PetscCheck(hip_status == rocblas_status_success, PETSC_COMM_SELF, PETSC_ERR_LIB, "rocsolver_*syevd failed for batch %" PetscInt_FMT, (PetscInt)i);
  }

  /* Check info */
  PetscCall(PetscMalloc1(n_batch, &h_info));
  PetscCallHIP(hipMemcpy(h_info, d_info, sizeof(int) * n_batch, hipMemcpyDeviceToHost));
  for (PetscInt i = 0; i < n_batch; i++) PetscCheck(h_info[i] == 0, PETSC_COMM_SELF, PETSC_ERR_LIB, "rocSOLVER eigendecomposition failed for matrix %" PetscInt_FMT ": info=%d", i, h_info[i]);
  PetscCall(PetscFree(h_info));

  /* Copy results back from contiguous layout to V_batch */
  Kokkos::parallel_for(
    "CopyResultsBack", Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, n_batch), KOKKOS_LAMBDA(const int i) {
      for (int j = 0; j < n_size; j++) {
        for (int k = 0; k < n_size; k++) V_batch(i, j, k) = d_A_contig[i * n_size * n_size + k * n_size + j];
        Lambda_batch(i, j) = d_W_contig[i * n_size + j];
      }
    });
  Kokkos::fence();
    #endif
  PetscFunctionReturn(PETSC_SUCCESS);
}
  #elif defined(KOKKOS_ENABLE_SYCL)
static PetscErrorCode BatchedEigenSolve_Device(LETKFView3D T_batch, LETKFView2D Lambda_batch, LETKFView3D V_batch, PetscInt n_batch, PetscInt n_size, sycl::queue *q, EigenWorkspace *work)
{
  PetscFunctionBegin;
    #if defined(PETSC_USE_COMPLEX)
  /* oneMKL's syevd USM overload targets real symmetric matrices; the complex analogue is heevd.
     The dispatcher gates the Kokkos path off when PETSC_USE_COMPLEX is set, so this is unreachable
     in practice; SETERRQ here as defense-in-depth in case that gate ever changes. */
  SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Complex numbers not supported on SYCL backend for LETKF");
    #else
  /* Use pre-allocated workspace */
  PetscScalar *d_work     = work->d_work;
  PetscScalar *d_A_contig = work->d_A_contig;
  PetscScalar *d_W_contig = work->d_W_contig;

  /* Copy T_batch to contiguous layout for oneMKL */
  Kokkos::parallel_for(
    "ReorganizeForOneMKL", Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, n_batch), KOKKOS_LAMBDA(const int i) {
      for (int j = 0; j < n_size; j++) {
        for (int k = 0; k < n_size; k++) d_A_contig[i * n_size * n_size + k * n_size + j] = T_batch(i, j, k);
      }
    });
  Kokkos::fence();

  /* oneMKL doesn't have a native batched syevd, so we loop over batch and call the USM
     overload of oneapi::mkl::lapack::syevd. The USM overload reports failures via
     oneapi::mkl::lapack::lapack_exception (a sycl::exception subclass), not through an
     output `info` parameter; PetscCallCXX() catches std::exception and converts to a
     PETSc error. */
  for (int i = 0; i < n_batch; i++) {
    PetscScalar *A_ptr = d_A_contig + i * n_size * n_size;
    PetscScalar *W_ptr = d_W_contig + i * n_size;

    PetscCallCXX(oneapi::mkl::lapack::syevd(*q, oneapi::mkl::job::vec, oneapi::mkl::uplo::upper, n_size, A_ptr, n_size, W_ptr, d_work, work->lwork_device));
    PetscCallCXX(q->wait_and_throw());
  }

  /* Copy results back from contiguous layout to V_batch */
  Kokkos::parallel_for(
    "CopyResultsBack", Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, n_batch), KOKKOS_LAMBDA(const int i) {
      for (int j = 0; j < n_size; j++) {
        for (int k = 0; k < n_size; k++) V_batch(i, j, k) = d_A_contig[i * n_size * n_size + k * n_size + j];
        Lambda_batch(i, j) = d_W_contig[i * n_size + j];
      }
    });
  Kokkos::fence();
    #endif
  PetscFunctionReturn(PETSC_SUCCESS);
}
  #endif
#endif

/*
  BatchedEigenSolve - Compute eigendecomposition for a batch of symmetric matrices

  Input Parameters:
+ T_batch      - batch of symmetric matrices (n_batch x n_size x n_size)
. n_batch      - number of matrices in the batch
- n_size       - size of each matrix (m x m)
- device_handle - device-specific solver handle (only for device builds)
- work         - reusable workspace structure

  Output Parameters:
+ Lambda_batch - eigenvalues for each matrix (n_batch x n_size)
- V_batch      - eigenvectors for each matrix (n_batch x n_size x n_size)

  Notes:
  Dispatcher function that calls the appropriate backend (Device or Host).
*/
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP) || defined(KOKKOS_ENABLE_SYCL)
  #if defined(KOKKOS_ENABLE_CUDA)
static PetscErrorCode BatchedEigenSolve(LETKFView3D T_batch, LETKFView2D Lambda_batch, LETKFView3D V_batch, PetscInt n_batch, PetscInt n_size, cusolverDnHandle_t device_handle, EigenWorkspace *work)
{
  PetscFunctionBegin;
  PetscCall(BatchedEigenSolve_Device(T_batch, Lambda_batch, V_batch, n_batch, n_size, device_handle, work));
  PetscFunctionReturn(PETSC_SUCCESS);
}
  #elif defined(KOKKOS_ENABLE_HIP)
static PetscErrorCode BatchedEigenSolve(LETKFView3D T_batch, LETKFView2D Lambda_batch, LETKFView3D V_batch, PetscInt n_batch, PetscInt n_size, rocblas_handle device_handle, EigenWorkspace *work)
{
  PetscFunctionBegin;
  PetscCall(BatchedEigenSolve_Device(T_batch, Lambda_batch, V_batch, n_batch, n_size, device_handle, work));
  PetscFunctionReturn(PETSC_SUCCESS);
}
  #elif defined(KOKKOS_ENABLE_SYCL)
static PetscErrorCode BatchedEigenSolve(LETKFView3D T_batch, LETKFView2D Lambda_batch, LETKFView3D V_batch, PetscInt n_batch, PetscInt n_size, sycl::queue *device_handle, EigenWorkspace *work)
{
  PetscFunctionBegin;
  PetscCall(BatchedEigenSolve_Device(T_batch, Lambda_batch, V_batch, n_batch, n_size, device_handle, work));
  PetscFunctionReturn(PETSC_SUCCESS);
}
  #endif
#else
static PetscErrorCode BatchedEigenSolve(LETKFView3D T_batch, LETKFView2D Lambda_batch, LETKFView3D V_batch, PetscInt n_batch, PetscInt n_size, EigenWorkspace *work)
{
  PetscFunctionBegin;
  PetscCall(BatchedEigenSolve_Host(T_batch, Lambda_batch, V_batch, n_batch, n_size, work));
  PetscFunctionReturn(PETSC_SUCCESS);
}
#endif

/*
  PetscDALETKFSetupLocalization_Kokkos - Prepares device views for localization matrix Q
*/
PETSC_INTERN PetscErrorCode PetscDALETKFSetupLocalization_Kokkos(PetscDA_LETKF *impl)
{
  PetscInt nrows, rstart, rend, i, nnz, total_nnz;

  PetscFunctionBegin;
  PetscCheck(impl->Q, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "impl->Q is not set; PetscDALETKFInstallQ() must run before SetupLocalization");
  PetscCall(PetscKokkosInitializeCheck());

  PetscCall(MatGetOwnershipRange(impl->Q, &rstart, &rend));
  nrows = rend - rstart;
  /* impl->n_nnz_local was populated by PetscDALETKFInstallQ() from MatGetInfo(MAT_LOCAL) so we
     don't re-query (which would force a device->host sync on AIJKOKKOS). */
  total_nnz = impl->n_nnz_local;

  /* Define View types */
  using view_1d_int    = Kokkos::View<PetscInt *, Kokkos::LayoutLeft>;
  using view_1d_scalar = Kokkos::View<PetscScalar *, Kokkos::LayoutLeft>;

  /* Allocate device views using actual total nnz from Q */
  view_1d_int    *d_Q_i;
  view_1d_int    *d_Q_j;
  view_1d_scalar *d_Q_a;

  PetscCallCXX(d_Q_i = new view_1d_int("Q_i", nrows + 1));
  PetscCallCXX(d_Q_j = new view_1d_int("Q_j", total_nnz));
  PetscCallCXX(d_Q_a = new view_1d_scalar("Q_a", total_nnz));

  /* Create host mirrors */
  Kokkos::View<PetscInt *, Kokkos::LayoutLeft, Kokkos::HostSpace>    h_Q_i;
  Kokkos::View<PetscInt *, Kokkos::LayoutLeft, Kokkos::HostSpace>    h_Q_j;
  Kokkos::View<PetscScalar *, Kokkos::LayoutLeft, Kokkos::HostSpace> h_Q_a;
  PetscCallCXX(h_Q_i = Kokkos::create_mirror_view(Kokkos::HostSpace(), *d_Q_i));
  PetscCallCXX(h_Q_j = Kokkos::create_mirror_view(Kokkos::HostSpace(), *d_Q_j));
  PetscCallCXX(h_Q_a = Kokkos::create_mirror_view(Kokkos::HostSpace(), *d_Q_a));

  /* Fill host mirrors with LOCAL indices into obs_work */
  h_Q_i(0) = 0;
  for (i = 0; i < nrows; i++) {
    const PetscInt    *cols;
    const PetscScalar *vals;
    PetscCall(MatGetRow(impl->Q, rstart + i, &nnz, &cols, &vals));
    h_Q_i(i + 1) = h_Q_i(i) + nnz;
    for (PetscInt k = 0; k < nnz; k++) {
      PetscInt local_idx;
      PetscCall(PetscHMapIGet(impl->obs_g2l, cols[k], &local_idx));
      PetscCheck(local_idx >= 0, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Observation index %" PetscInt_FMT " not found in local map", cols[k]);
      h_Q_j(h_Q_i(i) + k) = local_idx;
      h_Q_a(h_Q_i(i) + k) = vals[k];
    }
    PetscCall(MatRestoreRow(impl->Q, rstart + i, &nnz, &cols, &vals));
  }

  /* Copy to device */
  PetscCallCXX(Kokkos::deep_copy(*d_Q_i, h_Q_i));
  PetscCallCXX(Kokkos::deep_copy(*d_Q_j, h_Q_j));
  PetscCallCXX(Kokkos::deep_copy(*d_Q_a, h_Q_a));

  /* Store in impl */
  PetscCheck(!impl->Q_device_i, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "Q_device_i already allocated; PetscDALETKFDestroyLocalization_Kokkos must run before re-setup");
  impl->Q_device_i = static_cast<void *>(d_Q_i);
  impl->Q_device_j = static_cast<void *>(d_Q_j);
  impl->Q_device_a = static_cast<void *>(d_Q_a);
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  PetscDALETKFDestroyQDeviceMirrors_Kokkos - Free only the device-side CSR mirrors of Q.

  Used on the Q-rebuild path (setters that mutate type/radius/coordinates) so the persistent
  eigensolver workspace and the cusolver/rocblas/SYCL handle survive across rebuilds. The
  full destroy below also calls this helper.
*/
PETSC_INTERN PetscErrorCode PetscDALETKFDestroyQDeviceMirrors_Kokkos(PetscDA_LETKF *impl)
{
  PetscFunctionBegin;
  if (impl->Q_device_i) {
    using view_1d_int = Kokkos::View<PetscInt *, Kokkos::LayoutLeft>;
    delete static_cast<view_1d_int *>(impl->Q_device_i);
    impl->Q_device_i = NULL;
  }
  if (impl->Q_device_j) {
    using view_1d_int = Kokkos::View<PetscInt *, Kokkos::LayoutLeft>;
    delete static_cast<view_1d_int *>(impl->Q_device_j);
    impl->Q_device_j = NULL;
  }
  if (impl->Q_device_a) {
    using view_1d_scalar = Kokkos::View<PetscScalar *, Kokkos::LayoutLeft>;
    delete static_cast<view_1d_scalar *>(impl->Q_device_a);
    impl->Q_device_a = NULL;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  PetscDALETKFDestroyLocalization_Kokkos - Free all device-side state owned by the Kokkos backend.

  Tears down the Q device mirrors AND the persistent eigensolver workspace + cusolver/rocblas/SYCL
  handle. Both LOC_NONE (GlobalAnalysis_Kokkos) and the per-vertex paths allocate the latter
  state, so PetscDADestroy_LETKF calls this regardless of localization type. Q-rebuild paths
  use PetscDALETKFDestroyQDeviceMirrors_Kokkos() instead so the handle and workspace persist.
*/
PETSC_INTERN PetscErrorCode PetscDALETKFDestroyLocalization_Kokkos(PetscDA_LETKF *impl)
{
  PetscFunctionBegin;
  PetscCall(PetscDALETKFDestroyQDeviceMirrors_Kokkos(impl));

  /* Destroy solver handle and workspace */
  if (impl->eigen_work) {
    EigenWorkspace *work = static_cast<EigenWorkspace *>(impl->eigen_work);

#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP) || defined(KOKKOS_ENABLE_SYCL)
  #if defined(KOKKOS_ENABLE_CUDA)
    PetscCallCUDA(cudaFree(work->d_A_contig));
    PetscCallCUDA(cudaFree(work->d_W_contig));
    PetscCallCUDA(cudaFree(work->d_work));
    PetscCallCUDA(cudaFree(work->d_info));
    /* Destroy returns ignored: teardown may race with Kokkos/CUDA context shutdown when the
       enclosing PetscDA outlives PetscFinalize() handlers; raising here would mask the real
       teardown order issue. */
    if (work->syevj_params) cusolverDnDestroySyevjInfo(work->syevj_params);
  #elif defined(KOKKOS_ENABLE_HIP)
    PetscCallHIP(hipFree(work->d_A_contig));
    PetscCallHIP(hipFree(work->d_W_contig));
    PetscCallHIP(hipFree(work->d_work));
    PetscCallHIP(hipFree(work->d_info));
  #elif defined(KOKKOS_ENABLE_SYCL)
    if (impl->solver_handle) {
      sycl::queue *q = static_cast<sycl::queue *>(impl->solver_handle);
      if (work->d_A_contig) sycl::free(work->d_A_contig, *q);
      if (work->d_W_contig) sycl::free(work->d_W_contig, *q);
      if (work->d_work) sycl::free(work->d_work, *q);
      if (work->d_info) sycl::free(work->d_info, *q);
    }
  #endif
#else
  #if defined(PETSC_USE_COMPLEX)
    PetscCall(PetscFree4(work->all_v, work->all_lambda, work->all_work, work->all_rwork));
  #else
    PetscCall(PetscFree3(work->all_v, work->all_lambda, work->all_work));
  #endif
#endif

    delete work;
    impl->eigen_work = NULL;
  }

  if (impl->solver_handle) {
    /* Destroy returns ignored: see comment above on teardown-time race with Kokkos/CUDA
       context shutdown. */
#if defined(KOKKOS_ENABLE_CUDA)
    cusolverDnDestroy(static_cast<cusolverDnHandle_t>(impl->solver_handle));
#elif defined(KOKKOS_ENABLE_HIP)
    rocblas_destroy_handle(static_cast<rocblas_handle>(impl->solver_handle));
#elif defined(KOKKOS_ENABLE_SYCL)
    delete static_cast<sycl::queue *>(impl->solver_handle);
#endif
    impl->solver_handle = NULL;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* ========================================================================== */
/*                    LETKF Local Analysis (Main Function)                    */
/* ========================================================================== */

/*
  PetscDALETKFLocalAnalysis_Kokkos - Performs local LETKF analysis for all grid points (Kokkos version)

  Input Parameters:
+ da             - the PetscDA context
. impl           - LETKF implementation data
. m              - ensemble size
. n_vertices     - number of grid points
. X              - global anomaly matrix (state_size x m)
. observation    - observation vector
. Z_global       - global observation ensemble (obs_size x m)
. y_mean_global  - global observation mean
- r_inv_sqrt_global - global R^{-1/2}

  Output:
. da->ensemble - updated with analysis ensemble

  Notes:
  Kokkos device implementation of the LETKF local analysis. All n_vertices grid points are
  processed in a batched fashion: per-vertex S, T, V, T_sqrt, and weight slabs live in
  device 3-D/2-D views, observation data is mirrored from host when needed, and the
  per-vertex eigendecompositions are dispatched through BatchedEigenSolve_Device() (or
  BatchedEigenSolve_Host() when Kokkos's default execution space is the host).
*/
PETSC_INTERN PetscErrorCode PetscDALETKFLocalAnalysis_Kokkos(PetscDA da, PetscDA_LETKF *impl, PetscInt m, PetscInt n_vertices, Mat X, Vec observation, Mat Z_global, Vec y_mean_global, Vec r_inv_sqrt_global)
{
  using exec_space              = Kokkos::DefaultExecutionSpace;
  using view_3d                 = Kokkos::View<PetscScalar ***, Kokkos::LayoutLeft, exec_space>;
  using view_2d                 = Kokkos::View<PetscScalar **, Kokkos::LayoutLeft, exec_space>;
  using view_1d_int_const       = Kokkos::View<const PetscInt *, Kokkos::LayoutLeft, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
  using view_1d_scalar_const    = Kokkos::View<const PetscScalar *, Kokkos::LayoutLeft, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
  using view_1d_int             = Kokkos::View<PetscInt *, Kokkos::LayoutLeft>;
  using view_1d_scalar          = Kokkos::View<PetscScalar *, Kokkos::LayoutLeft>;
  using view_2d_unmanaged       = Kokkos::View<const PetscScalar **, Kokkos::LayoutLeft, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
  using view_1d_unmanaged       = Kokkos::View<const PetscScalar *, Kokkos::LayoutLeft, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
  using view_2d_unmanaged_write = Kokkos::View<PetscScalar **, Kokkos::LayoutLeft, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
  PetscDA_Ensemble       *en    = &impl->en;
  EigenWorkspace         *eigen_work;
  PetscInt                ndof;
  PetscInt                lda_z_global, lda_x, lda_e, n_obs_local;
  PetscInt                max_nnz_per_row, max_nnz_copy, chunk_size;
  PetscInt64              mem_per_point;
  PetscReal               sqrt_m_minus_1, scale, inflation_inv;
  PetscReal               flops, n_obs_total;
  PetscMemType            z_mem_type, y_mem_type, y_mean_mem_type, r_inv_sqrt_mem_type;
  PetscMemType            x_mem_type, mean_mem_type, e_mem_type;
  const PetscScalar      *z_global_array, *y_global_array, *y_mean_global_array, *r_inv_sqrt_global_array;
  const PetscScalar      *x_array, *mean_array;
  PetscScalar            *e_array;
  const PetscScalar      *z_ptr, *y_ptr, *y_mean_ptr, *r_inv_sqrt_ptr;
  const PetscScalar      *x_ptr, *mean_ptr;
  PetscScalar            *e_ptr;
  PetscBool               e_is_copy = PETSC_FALSE;
  view_1d_int_const       Q_i_view, Q_j_view;
  view_1d_scalar_const    Q_a_view;
  LETKFView2D             z_managed, x_managed, e_managed;
  LETKFView1D             y_managed, y_mean_managed, r_inv_sqrt_managed, mean_managed;
  view_2d_unmanaged       Z_global_view, X_view;
  view_1d_unmanaged       y_global_view, y_mean_global_view, r_inv_sqrt_global_view, mean_view;
  view_2d_unmanaged_write E_view;
  view_3d                 S_batch, T_batch, V_batch, T_sqrt_batch;
  view_2d                 Lambda_batch, w_batch, delta_batch, y_batch, y_mean_batch, r_inv_sqrt_batch, temp1_batch, temp2_batch, inv_sqrt_lambda_batch;
#if defined(KOKKOS_ENABLE_CUDA)
  cusolverDnHandle_t device_handle = nullptr;
  cusolverStatus_t   cusolver_status;
#elif defined(KOKKOS_ENABLE_HIP)
  rocblas_handle device_handle = nullptr;
#elif defined(KOKKOS_ENABLE_SYCL)
  sycl::queue *device_handle = nullptr;
#endif

  PetscFunctionBegin;
  ndof           = da->ndof;
  scale          = 1.0 / PetscSqrtReal((PetscReal)(m - 1));
  sqrt_m_minus_1 = PetscSqrtReal((PetscReal)(m - 1));
  inflation_inv  = 1.0 / en->inflation; /* (1/rho) for T matrix: T = (1/rho)I + S^T*S */

  /* ===================================================================== */
  /* Step 2.1.1: Create batched workspace for ALL grid points            */
  /* ===================================================================== */
  /*
     NOTE ON PARALLELISM STRATEGY:
     We use Kokkos::RangePolicy over grid points (n_vertices) combined with KokkosBatched::Serial kernels.
     Since the data layout is LayoutLeft (Column-Major) to match PETSc/LAPACK, the index 'i' (grid point)
     is the fastest varying index (stride 1).

     RangePolicy maps consecutive threads to consecutive 'i', ensuring perfect memory coalescing
     when accessing arrays like S_batch(i, p, j).

     Using TeamPolicy/TeamVectorRange to parallelize inner loops (m or p) would assign a team to 'i',
     causing threads within the team to access S_batch with stride 'n_vertices', which leads to
     uncoalesced memory access and poor performance on GPUs.

     Therefore, RangePolicy + SerialGemm is the optimal strategy for this data layout.
  */

  /* ===================================================================== */
  /* Step 2.1.2a: Pre-extract Q matrix CSR data for device access        */
  /* ===================================================================== */
  PetscCheck(impl->Q_device_i, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "Q device views not allocated; PetscDALETKFSetupLocalization_Kokkos must run before LocalAnalysis");
  /* Use pre-allocated device views */
  view_1d_int    *d_Q_i = static_cast<view_1d_int *>(impl->Q_device_i);
  view_1d_int    *d_Q_j = static_cast<view_1d_int *>(impl->Q_device_j);
  view_1d_scalar *d_Q_a = static_cast<view_1d_scalar *>(impl->Q_device_a);

  Q_i_view = view_1d_int_const(d_Q_i->data(), d_Q_i->extent(0));
  Q_j_view = view_1d_int_const(d_Q_j->data(), d_Q_j->extent(0));
  Q_a_view = view_1d_scalar_const(d_Q_a->data(), d_Q_a->extent(0));

  /* Get global observation data arrays */
  PetscCall(MatDenseGetArrayReadAndMemType(Z_global, &z_global_array, &z_mem_type));
  PetscCall(VecGetArrayReadAndMemType(observation, &y_global_array, &y_mem_type));
  PetscCall(VecGetArrayReadAndMemType(y_mean_global, &y_mean_global_array, &y_mean_mem_type));
  PetscCall(VecGetArrayReadAndMemType(r_inv_sqrt_global, &r_inv_sqrt_global_array, &r_inv_sqrt_mem_type));
  PetscCall(MatDenseGetLDA(Z_global, &lda_z_global));
  PetscCall(VecGetLocalSize(observation, &n_obs_local));

  /* Handle memory mirroring for observation data. The 1-D obs vectors are sized n_obs_local;
     only the 2-D Z view is shaped by lda_z_global. */
  z_ptr          = z_global_array;
  y_ptr          = y_global_array;
  y_mean_ptr     = y_mean_global_array;
  r_inv_sqrt_ptr = r_inv_sqrt_global_array;

  if (z_mem_type == PETSC_MEMTYPE_HOST) {
    Kokkos::View<const PetscScalar **, Kokkos::LayoutLeft, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> src(z_global_array, lda_z_global, m);
    PetscCallCXX(z_managed = LETKFView2D("z_managed", lda_z_global, m));
    PetscCallCXX(Kokkos::deep_copy(z_managed, src));
    z_ptr = z_managed.data();
  }
  if (y_mem_type == PETSC_MEMTYPE_HOST) {
    Kokkos::View<const PetscScalar *, Kokkos::LayoutLeft, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> src(y_global_array, n_obs_local);
    PetscCallCXX(y_managed = LETKFView1D("y_managed", n_obs_local));
    PetscCallCXX(Kokkos::deep_copy(y_managed, src));
    y_ptr = y_managed.data();
  }
  if (y_mean_mem_type == PETSC_MEMTYPE_HOST) {
    Kokkos::View<const PetscScalar *, Kokkos::LayoutLeft, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> src(y_mean_global_array, n_obs_local);
    PetscCallCXX(y_mean_managed = LETKFView1D("y_mean_managed", n_obs_local));
    PetscCallCXX(Kokkos::deep_copy(y_mean_managed, src));
    y_mean_ptr = y_mean_managed.data();
  }
  if (r_inv_sqrt_mem_type == PETSC_MEMTYPE_HOST) {
    Kokkos::View<const PetscScalar *, Kokkos::LayoutLeft, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> src(r_inv_sqrt_global_array, n_obs_local);
    PetscCallCXX(r_inv_sqrt_managed = LETKFView1D("r_inv_sqrt_managed", n_obs_local));
    PetscCallCXX(Kokkos::deep_copy(r_inv_sqrt_managed, src));
    r_inv_sqrt_ptr = r_inv_sqrt_managed.data();
  }

  /* Create unmanaged Kokkos views for global observation data. NOTE: Z_global_view's leading
     extent is lda_z_global (the MatDense column stride) while the 1-D obs views use n_obs_local
     (the unpadded local-row count). Always index Z_global_view as Z_global_view(obs_idx, j) with
     obs_idx < n_obs_local taken from Q's per-row column list; never iterate [0, extent(0)) on Z
     because that walks into LDA padding. */
  Z_global_view          = view_2d_unmanaged(z_ptr, lda_z_global, m);
  y_global_view          = view_1d_unmanaged(y_ptr, n_obs_local);
  y_mean_global_view     = view_1d_unmanaged(y_mean_ptr, n_obs_local);
  r_inv_sqrt_global_view = view_1d_unmanaged(r_inv_sqrt_ptr, n_obs_local);

  /* Get access to global X matrix and mean vector */
  PetscCall(MatDenseGetArrayReadAndMemType(X, &x_array, &x_mem_type));
  PetscCall(VecGetArrayReadAndMemType(impl->mean, &mean_array, &mean_mem_type));
  PetscCall(MatDenseGetArrayWriteAndMemType(en->ensemble, &e_array, &e_mem_type));
  PetscCall(MatDenseGetLDA(X, &lda_x));
  PetscCall(MatDenseGetLDA(en->ensemble, &lda_e));
  /* Per-vertex subviews into X/E span [i_global*ndof, (i_global+1)*ndof); the maximum global
     index is n_vertices-1, so the leading dimension must cover all local vertices. */
  PetscCheck(lda_x >= n_vertices * ndof, PetscObjectComm((PetscObject)X), PETSC_ERR_ARG_INCOMP, "X leading dimension %" PetscInt_FMT " < n_vertices*ndof %" PetscInt_FMT, lda_x, n_vertices * ndof);
  PetscCheck(lda_e >= n_vertices * ndof, PetscObjectComm((PetscObject)en->ensemble), PETSC_ERR_ARG_INCOMP, "Ensemble leading dimension %" PetscInt_FMT " < n_vertices*ndof %" PetscInt_FMT, lda_e, n_vertices * ndof);

  /* Handle memory mirroring for state data */
  x_ptr    = x_array;
  mean_ptr = mean_array;
  e_ptr    = e_array;

  if (x_mem_type == PETSC_MEMTYPE_HOST) {
    Kokkos::View<const PetscScalar **, Kokkos::LayoutLeft, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> src(x_array, lda_x, m);
    PetscCallCXX(x_managed = LETKFView2D("x_managed", lda_x, m));
    PetscCallCXX(Kokkos::deep_copy(x_managed, src));
    x_ptr = x_managed.data();
  }
  if (mean_mem_type == PETSC_MEMTYPE_HOST) {
    /* impl->mean is a Vec with local size n_vertices*ndof (no MatDense LDA padding), so size the
       mirror to the exact buffer extent; reading lda_x would over-read when MatDense pads X's LDA. */
    Kokkos::View<const PetscScalar *, Kokkos::LayoutLeft, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> src(mean_array, n_vertices * ndof);
    PetscCallCXX(mean_managed = LETKFView1D("mean_managed", n_vertices * ndof));
    PetscCallCXX(Kokkos::deep_copy(mean_managed, src));
    mean_ptr = mean_managed.data();
  }
  if (e_mem_type == PETSC_MEMTYPE_HOST) {
    Kokkos::View<PetscScalar **, Kokkos::LayoutLeft, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> src(e_array, lda_e, m);
    PetscCallCXX(e_managed = LETKFView2D("e_managed", lda_e, m));
    PetscCallCXX(Kokkos::deep_copy(e_managed, src));
    e_ptr     = e_managed.data();
    e_is_copy = PETSC_TRUE;
  }

  /* Create unmanaged Kokkos views for global data */
  X_view    = view_2d_unmanaged(const_cast<PetscScalar *>(x_ptr), lda_x, m);
  mean_view = view_1d_unmanaged(mean_ptr, n_vertices * ndof);
  E_view    = view_2d_unmanaged_write(e_ptr, lda_e, m);

  max_nnz_per_row = impl->max_nnz_per_row;

  /* Determine chunk size to avoid OOM on large grids */
  mem_per_point = (PetscInt64)sizeof(PetscScalar) * ((PetscInt64)m * m + (PetscInt64)max_nnz_per_row * m);
  if (impl->batch_size > 0) {
    chunk_size = impl->batch_size;
  } else {
    /* Target ~2GB workspace. Approx memory per point: m*m*sizeof(PetscScalar) (T) + p*m*sizeof(PetscScalar) (Z). */
    chunk_size = (PetscInt)((PetscInt64)2 * 1024 * 1024 * 1024 / mem_per_point);
    /* Clamp to reasonable max to avoid huge allocations even if memory allows */
    if (chunk_size > 32768) chunk_size = 32768;
  }

  if (chunk_size < 1) chunk_size = 1;
  if (chunk_size > n_vertices) chunk_size = n_vertices;

  /* OPTIMIZATION: Create device solver handle once, reuse across chunks */
#if defined(KOKKOS_ENABLE_CUDA)
  if (impl->solver_handle) device_handle = static_cast<cusolverDnHandle_t>(impl->solver_handle);
  else {
    cusolver_status = cusolverDnCreate(&device_handle);
    PetscCheck(cusolver_status == CUSOLVER_STATUS_SUCCESS, PETSC_COMM_SELF, PETSC_ERR_LIB, "cusolverDnCreate failed");
    impl->solver_handle = static_cast<void *>(device_handle);
  }
#elif defined(KOKKOS_ENABLE_HIP)
  if (impl->solver_handle) device_handle = static_cast<rocblas_handle>(impl->solver_handle);
  else {
    rocblas_status hip_status = rocblas_create_handle(&device_handle);
    PetscCheck(hip_status == rocblas_status_success, PETSC_COMM_SELF, PETSC_ERR_LIB, "rocblas_create_handle failed");
    impl->solver_handle = static_cast<void *>(device_handle);
  }
#elif defined(KOKKOS_ENABLE_SYCL)
  if (impl->solver_handle) device_handle = static_cast<sycl::queue *>(impl->solver_handle);
  else {
    PetscCallCXX(device_handle = new sycl::queue(sycl::gpu_selector_v));
    impl->solver_handle = static_cast<void *>(device_handle);
  }
#endif

  /* ===================================================================== */
  /* OPTIMIZATION: Hoist allocations outside the chunk loop                */
  /* ===================================================================== */
  /* Allocate Kokkos Views once for the maximum chunk size */
  max_nnz_copy = max_nnz_per_row;

  eigen_work = static_cast<EigenWorkspace *>(impl->eigen_work);
  if (!eigen_work) {
    PetscCallCXX(eigen_work = new EigenWorkspace());
    impl->eigen_work = static_cast<void *>(eigen_work);
  }

  /* Check if reallocation is needed */
  if (eigen_work->max_chunk_size < chunk_size || eigen_work->m != m || eigen_work->max_nnz < max_nnz_copy) {
    /* Free old device workspace if exists */
#if defined(KOKKOS_ENABLE_CUDA)
    PetscCallCUDA(cudaFree(eigen_work->d_work));
    PetscCallCUDA(cudaFree(eigen_work->d_info));
    PetscCallCUDA(cudaFree(eigen_work->d_A_contig));
    PetscCallCUDA(cudaFree(eigen_work->d_W_contig));
    if (eigen_work->syevj_params) cusolverDnDestroySyevjInfo(eigen_work->syevj_params);
    eigen_work->syevj_params = nullptr;
#elif defined(KOKKOS_ENABLE_HIP)
    PetscCallHIP(hipFree(eigen_work->d_work));
    PetscCallHIP(hipFree(eigen_work->d_info));
    PetscCallHIP(hipFree(eigen_work->d_A_contig));
    PetscCallHIP(hipFree(eigen_work->d_W_contig));
#elif defined(KOKKOS_ENABLE_SYCL)
    if (eigen_work->d_work) sycl::free(eigen_work->d_work, *device_handle);
    if (eigen_work->d_info) sycl::free(eigen_work->d_info, *device_handle);
    if (eigen_work->d_A_contig) sycl::free(eigen_work->d_A_contig, *device_handle);
    if (eigen_work->d_W_contig) sycl::free(eigen_work->d_W_contig, *device_handle);
#endif

#if !defined(KOKKOS_ENABLE_CUDA) && !defined(KOKKOS_ENABLE_HIP) && !defined(KOKKOS_ENABLE_SYCL)
  #if defined(PETSC_USE_COMPLEX)
    if (eigen_work->all_v) PetscCall(PetscFree4(eigen_work->all_v, eigen_work->all_lambda, eigen_work->all_work, eigen_work->all_rwork));
  #else
    if (eigen_work->all_v) PetscCall(PetscFree3(eigen_work->all_v, eigen_work->all_lambda, eigen_work->all_work));
  #endif
#endif

    /* Update dimensions */
    eigen_work->max_chunk_size = chunk_size;
    eigen_work->m              = m;
    eigen_work->max_nnz        = max_nnz_copy;

    /* Allocate Kokkos Views */
    PetscCallCXX(eigen_work->S_batch = view_3d("S_batch", chunk_size, max_nnz_copy, m));
    PetscCallCXX(eigen_work->T_batch = view_3d("T_batch", chunk_size, m, m));
    /* Alias: the eigensolve overwrites T in place, so V and T share storage. Any future
       kernel that needs the original symmetric T after the eigensolve must allocate V
       separately (view_3d("V_batch", chunk_size, m, m)) instead of aliasing. */
    eigen_work->V_batch = eigen_work->T_batch;
    PetscCallCXX(eigen_work->Lambda_batch = view_2d("Lambda_batch", chunk_size, m));
    PetscCallCXX(eigen_work->T_sqrt_batch = view_3d("T_sqrt_batch", chunk_size, m, m));
    PetscCallCXX(eigen_work->w_batch = view_2d("w_batch", chunk_size, m));
    PetscCallCXX(eigen_work->delta_batch = view_2d("delta_batch", chunk_size, max_nnz_copy));
    PetscCallCXX(eigen_work->y_batch = view_2d("y_batch", chunk_size, max_nnz_copy));
    PetscCallCXX(eigen_work->y_mean_batch = view_2d("y_mean_batch", chunk_size, max_nnz_copy));
    PetscCallCXX(eigen_work->r_inv_sqrt_batch = view_2d("r_inv_sqrt_batch", chunk_size, max_nnz_copy));
    PetscCallCXX(eigen_work->temp1_batch = view_2d("temp1_batch", chunk_size, m));
    PetscCallCXX(eigen_work->temp2_batch = view_2d("temp2_batch", chunk_size, m));
    PetscCallCXX(eigen_work->inv_sqrt_lambda_batch = view_2d("inv_sqrt_lambda_batch", chunk_size, m));

    /* Allocate solver workspace */
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP) || defined(KOKKOS_ENABLE_SYCL)
  #if defined(KOKKOS_ENABLE_CUDA)
    {
      /* Create syevj params */
      cusolver_status = cusolverDnCreateSyevjInfo(&eigen_work->syevj_params);
      PetscCheck(cusolver_status == CUSOLVER_STATUS_SUCCESS, PETSC_COMM_SELF, PETSC_ERR_LIB, "cusolverDnCreateSyevjInfo failed");

      /* Set default params */
      cusolver_status = cusolverDnXsyevjSetTolerance(eigen_work->syevj_params, 1e-7);
      PetscCheck(cusolver_status == CUSOLVER_STATUS_SUCCESS, PETSC_COMM_SELF, PETSC_ERR_LIB, "cusolverDnXsyevjSetTolerance failed");
      cusolver_status = cusolverDnXsyevjSetMaxSweeps(eigen_work->syevj_params, 100);
      PetscCheck(cusolver_status == CUSOLVER_STATUS_SUCCESS, PETSC_COMM_SELF, PETSC_ERR_LIB, "cusolverDnXsyevjSetMaxSweeps failed");
      cusolver_status = cusolverDnXsyevjSetSortEig(eigen_work->syevj_params, 1); /* Sort eigenvalues */
      PetscCheck(cusolver_status == CUSOLVER_STATUS_SUCCESS, PETSC_COMM_SELF, PETSC_ERR_LIB, "cusolverDnXsyevjSetSortEig failed");

      /* Query workspace size */
      PetscScalar *d_A = eigen_work->T_batch.data();
      PetscScalar *d_W = eigen_work->Lambda_batch.data();
      int          lwork;
    #if defined(PETSC_USE_REAL_SINGLE)
      cusolver_status = cusolverDnSsyevjBatched_bufferSize(device_handle, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_UPPER, m, d_A, m, d_W, &lwork, eigen_work->syevj_params, chunk_size);
    #else
      cusolver_status = cusolverDnDsyevjBatched_bufferSize(device_handle, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_UPPER, m, d_A, m, d_W, &lwork, eigen_work->syevj_params, chunk_size);
    #endif
      PetscCheck(cusolver_status == CUSOLVER_STATUS_SUCCESS, PETSC_COMM_SELF, PETSC_ERR_LIB, "cusolverDn*syevjBatched_bufferSize failed");
      eigen_work->lwork_device = lwork;

      /* Allocate workspace */
      PetscCallCUDA(cudaMalloc(&eigen_work->d_work, sizeof(PetscScalar) * lwork));
      PetscCallCUDA(cudaMalloc(&eigen_work->d_info, sizeof(int) * chunk_size));
      PetscCallCUDA(cudaMalloc(&eigen_work->d_A_contig, sizeof(PetscScalar) * chunk_size * m * m));
      PetscCallCUDA(cudaMalloc(&eigen_work->d_W_contig, sizeof(PetscScalar) * chunk_size * m));
    }
  #elif defined(KOKKOS_ENABLE_HIP)
    {
        /* rocsolver_*syevd takes a single n-element off-diagonal scratch buffer (the E array).
         The batch loop is sequential, so one shared buffer is sufficient. */
    #if defined(PETSC_USE_COMPLEX)
      int lwork = 0; /* Complex not supported on device */
    #else
      PetscCheck(m <= INT_MAX, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Ensemble size m=%" PetscInt_FMT " exceeds INT_MAX for rocsolver lwork", m);
      int lwork = (int)m;
    #endif
      eigen_work->lwork_device = lwork;

      /* Allocate workspace */
      if (lwork > 0) {
        PetscCallHIP(hipMalloc(&eigen_work->d_work, sizeof(PetscScalar) * lwork));
        PetscCallHIP(hipMalloc(&eigen_work->d_info, sizeof(int) * chunk_size));
        PetscCallHIP(hipMalloc(&eigen_work->d_A_contig, sizeof(PetscScalar) * chunk_size * m * m));
        PetscCallHIP(hipMalloc(&eigen_work->d_W_contig, sizeof(PetscScalar) * chunk_size * m));
      }
    }
  #elif defined(KOKKOS_ENABLE_SYCL)
    {
      /* Query the exact scratchpad size oneMKL needs for syevd. The hand-rolled formula that
         used to live here was guessed from textbook LAPACK requirements and is not guaranteed
         to match every oneMKL backend. */
      std::int64_t lwork = 0;
      PetscCallCXX(lwork = oneapi::mkl::lapack::syevd_scratchpad_size<PetscScalar>(*device_handle, oneapi::mkl::job::vec, oneapi::mkl::uplo::upper, m, m));
      PetscCheck(lwork <= (std::int64_t)INT_MAX, PETSC_COMM_SELF, PETSC_ERR_PLIB, "oneMKL syevd_scratchpad_size %lld exceeds INT_MAX", (long long)lwork);
      eigen_work->lwork_device = (int)lwork;

      /* Allocate workspace using SYCL malloc_device. The USM overload of syevd reports failures
         via lapack_exception, not an info argument, so d_info is not allocated. */
      eigen_work->d_work     = sycl::malloc_device<PetscScalar>(lwork, *device_handle);
      eigen_work->d_A_contig = sycl::malloc_device<PetscScalar>(chunk_size * m * m, *device_handle);
      eigen_work->d_W_contig = sycl::malloc_device<PetscScalar>(chunk_size * m, *device_handle);
      eigen_work->d_info     = nullptr;
      PetscCheck(eigen_work->d_work && eigen_work->d_A_contig && eigen_work->d_W_contig, PETSC_COMM_SELF, PETSC_ERR_MEM, "SYCL memory allocation failed");
    }
  #endif
#else
    {
      PetscBLASInt n_blas;
      PetscCall(PetscBLASIntCast(m, &n_blas));
      eigen_work->n_blas = n_blas;

      /* Query workspace size */
      PetscBLASInt lwork_query = -1;
      PetscScalar  work_query;
      PetscBLASInt info;
  #if defined(PETSC_USE_COMPLEX)
      PetscReal rwork_query;
      LAPACKsyev_("V", "U", &n_blas, &work_query, &n_blas, &rwork_query, &work_query, &lwork_query, &rwork_query, &info);
  #else
      LAPACKsyev_("V", "U", &n_blas, &work_query, &n_blas, &work_query, &work_query, &lwork_query, &info);
  #endif
      PetscCheck(info == 0, PETSC_COMM_SELF, PETSC_ERR_LIB, "LAPACK workspace query failed on SYEV %" PetscBLASInt_FMT, info);
      eigen_work->lwork = (PetscBLASInt)PetscRealPart(work_query);

      /* Allocate workspace */
  #if defined(PETSC_USE_COMPLEX)
      PetscCall(PetscMalloc4(chunk_size * m * m, &eigen_work->all_v, chunk_size * m, &eigen_work->all_lambda, chunk_size * eigen_work->lwork, &eigen_work->all_work, chunk_size * (3 * m - 2), &eigen_work->all_rwork));
  #else
      PetscCall(PetscMalloc3(chunk_size * m * m, &eigen_work->all_v, chunk_size * m, &eigen_work->all_lambda, chunk_size * eigen_work->lwork, &eigen_work->all_work));
  #endif
    }
#endif
  }

  /* Local aliases so KOKKOS_LAMBDAs capture views by value, not via eigen_work-> */
  S_batch               = eigen_work->S_batch;
  T_batch               = eigen_work->T_batch;
  V_batch               = eigen_work->V_batch;
  Lambda_batch          = eigen_work->Lambda_batch;
  T_sqrt_batch          = eigen_work->T_sqrt_batch;
  w_batch               = eigen_work->w_batch;
  delta_batch           = eigen_work->delta_batch;
  y_batch               = eigen_work->y_batch;
  y_mean_batch          = eigen_work->y_mean_batch;
  r_inv_sqrt_batch      = eigen_work->r_inv_sqrt_batch;
  temp1_batch           = eigen_work->temp1_batch;
  temp2_batch           = eigen_work->temp2_batch;
  inv_sqrt_lambda_batch = eigen_work->inv_sqrt_lambda_batch;

  /* Loop over chunks */
  for (PetscInt chunk_start = 0; chunk_start < n_vertices; chunk_start += chunk_size) {
    PetscInt chunk_end       = (chunk_start + chunk_size > n_vertices) ? n_vertices : chunk_start + chunk_size;
    PetscInt n_batch_current = chunk_end - chunk_start;

    /* No pre-zeroing of S_batch/delta_batch/y_*_batch/r_inv_sqrt_batch is required: the
       fused extractor writes positions [0, ncols) on every iteration, and every downstream
       consumer (ComputeAllTMatrices, ComputeWeightsAndInvSqrtLambda, the DEBUG NaN check)
       reads through subviews bounded by the per-row ncols. Stale values in [ncols, max_nnz_per_row)
       carried over from a previous chunk are unreachable. */

    /* ===================================================================== */
    /* Step 2.1.2: Fused observation extraction and S/Delta computation     */
    /* ===================================================================== */
    /* Extract local observations and immediately compute S and delta       */
    /* This fusion eliminates one kernel launch and improves cache locality */
    Kokkos::parallel_for(
      "ExtractAndComputeSAndDelta", Kokkos::RangePolicy<exec_space>(0, n_batch_current), KOKKOS_LAMBDA(const int i_local) {
        PetscInt i_global = chunk_start + i_local;
        /* Get Q row for this grid point using CSR format */
        PetscInt row_start = Q_i_view(i_global);
        PetscInt row_end   = Q_i_view(i_global + 1);
        PetscInt ncols     = row_end - row_start;

        /* Extract observations and compute S/delta for this grid point */
        for (PetscInt k = 0; k < ncols; k++) {
          PetscInt    obs_idx = Q_j_view(row_start + k);
          PetscScalar weight  = Q_a_view(row_start + k);

          /* Extract observation vectors */
          PetscScalar y_val      = y_global_view(obs_idx);
          PetscScalar y_mean_val = y_mean_global_view(obs_idx);
          PetscScalar r_inv_sqrt = r_inv_sqrt_global_view(obs_idx) * Kokkos::sqrt(PetscRealPart(weight));

          /* Store for later use if needed */
          y_batch(i_local, k)          = y_val;
          y_mean_batch(i_local, k)     = y_mean_val;
          r_inv_sqrt_batch(i_local, k) = r_inv_sqrt;

          /* Compute delta immediately: delta = R^{-1/2}(y - y_mean) */
          delta_batch(i_local, k) = (y_val - y_mean_val) * r_inv_sqrt;

          /* Compute S row: S = R^{-1/2}(Z - y_mean * 1')/sqrt(m-1) */
          PetscScalar scale_factor = scale * r_inv_sqrt;
          for (int j = 0; j < m; j++) S_batch(i_local, k, j) = (Z_global_view(obs_idx, j) - y_mean_val) * scale_factor;
        }
      });
    Kokkos::fence();

    /* DEBUG: Check S for NaNs */
    if (PetscDefined(USE_DEBUG)) {
      PetscInt nan_count = 0;
      Kokkos::parallel_reduce(
        "CheckS", Kokkos::RangePolicy<exec_space>(0, n_batch_current),
        KOKKOS_LAMBDA(const int i, PetscInt &l_count) {
          PetscInt i_global = chunk_start + i;
          PetscInt ncols    = Q_i_view(i_global + 1) - Q_i_view(i_global);
          for (PetscInt j = 0; j < ncols; j++) {
            for (int k = 0; k < m; k++) {
              if (S_batch(i, j, k) != S_batch(i, j, k)) l_count++;
            }
          }
        },
        nan_count);
      PetscCheck(nan_count == 0, PETSC_COMM_SELF, PETSC_ERR_FP, "Found %" PetscInt_FMT " NaNs in S_batch at chunk_start %" PetscInt_FMT, nan_count, chunk_start);
    }

    /* ===================================================================== */
    /* Step 2.1.4: Optimized T matrix formation (T = (1/rho)I + S^T * S)    */
    /* ===================================================================== */
    /* Compute T_i = (1/rho)I + S_i^T * S_i for current chunk */
    /* Exploit symmetry: only compute upper triangle, then copy to lower */
    /* This reduces operations by ~50% */
    Kokkos::parallel_for(
      "ComputeAllTMatrices", Kokkos::RangePolicy<exec_space>(0, n_batch_current), KOKKOS_LAMBDA(const int i) {
        PetscInt i_global = chunk_start + i;
        PetscInt ncols    = Q_i_view(i_global + 1) - Q_i_view(i_global);

        /* Compute upper triangle of T_i = (1/rho)I + S_i^T * S_i */
        /* T_i(j,k) = (1/rho)*delta_jk + sum_p S_i(p,j) * S_i(p,k) for j <= k */
        for (int j = 0; j < m; j++) {
          for (int k = j; k < m; k++) {
            PetscScalar sum = (j == k) ? inflation_inv : 0.0;
            for (PetscInt p = 0; p < ncols; p++) sum += S_batch(i, p, j) * S_batch(i, p, k);
            T_batch(i, j, k) = sum;
          }
        }

        /* Copy upper triangle to lower triangle (T is symmetric) */
        for (int j = 0; j < m; j++) {
          for (int k = 0; k < j; k++) T_batch(i, j, k) = T_batch(i, k, j);
        }
      });
    Kokkos::fence();

    /* DEBUG: Check T for NaNs */
    if (PetscDefined(USE_DEBUG)) {
      PetscInt nan_count = 0;
      Kokkos::parallel_reduce(
        "CheckT", Kokkos::RangePolicy<exec_space>(0, n_batch_current),
        KOKKOS_LAMBDA(const int i, PetscInt &l_count) {
          for (int j = 0; j < m; j++) {
            for (int k = 0; k < m; k++) {
              if (T_batch(i, j, k) != T_batch(i, j, k)) l_count++;
            }
          }
        },
        nan_count);
      PetscCheck(nan_count == 0, PETSC_COMM_SELF, PETSC_ERR_FP, "Found %" PetscInt_FMT " NaNs in T_batch at chunk_start %" PetscInt_FMT, nan_count, chunk_start);
    }

    /* ===================================================================== */
    /* Step 3.1.1: Batched eigendecomposition for current chunk            */
    /* ===================================================================== */
    /* Compute T_i = V_i * Lambda_i * V_i^T for current chunk */
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP) || defined(KOKKOS_ENABLE_SYCL)
    PetscCall(BatchedEigenSolve(T_batch, Lambda_batch, V_batch, n_batch_current, m, device_handle, eigen_work));
#else
    PetscCall(BatchedEigenSolve(T_batch, Lambda_batch, V_batch, n_batch_current, m, eigen_work));
#endif

    /* DEBUG: Check Lambda for NaNs or negative values */
    if (PetscDefined(USE_DEBUG)) {
      PetscInt bad_lambda = 0;
      Kokkos::parallel_reduce(
        "CheckLambda", Kokkos::RangePolicy<exec_space>(0, n_batch_current),
        KOKKOS_LAMBDA(const int i, PetscInt &l_count) {
          for (int k = 0; k < m; k++) {
            if (Lambda_batch(i, k) != Lambda_batch(i, k) || PetscRealPart(Lambda_batch(i, k)) < -1e-8) l_count++;
          }
        },
        bad_lambda);
      PetscCheck(bad_lambda == 0, PETSC_COMM_SELF, PETSC_ERR_FP, "Found %" PetscInt_FMT " bad eigenvalues (NaN or negative) at chunk_start %" PetscInt_FMT, bad_lambda, chunk_start);
    }

    /* ===================================================================== */
    /* Step 3.1.2: Precompute w and inv_sqrt_lambda for ensemble update    */
    /* ===================================================================== */
    /* Compute w_i = T_i^{-1} * (S_i^T * delta_i) using eigendecomposition */
    /* Precompute 1/sqrt(Lambda) for use in ensemble update */
    Kokkos::parallel_for(
      "ComputeWeightsAndInvSqrtLambda", Kokkos::RangePolicy<exec_space>(0, n_batch_current), KOKKOS_LAMBDA(const int i) {
        PetscInt i_global          = chunk_start + i;
        PetscInt ncols             = Q_i_view(i_global + 1) - Q_i_view(i_global);
        auto     S_i               = Kokkos::subview(S_batch, i, Kokkos::make_pair((PetscInt)0, ncols), Kokkos::ALL());
        auto     V_i               = Kokkos::subview(V_batch, i, Kokkos::ALL(), Kokkos::ALL());
        auto     Lambda_i          = Kokkos::subview(Lambda_batch, i, Kokkos::ALL());
        auto     delta_i           = Kokkos::subview(delta_batch, i, Kokkos::make_pair((PetscInt)0, ncols));
        auto     w_i               = Kokkos::subview(w_batch, i, Kokkos::ALL());
        auto     inv_sqrt_lambda_i = Kokkos::subview(inv_sqrt_lambda_batch, i, Kokkos::ALL());
        auto     temp1             = Kokkos::subview(temp1_batch, i, Kokkos::ALL());
        auto     temp2             = Kokkos::subview(temp2_batch, i, Kokkos::ALL());

        /* 1. Compute w_i = V * L^-1 * V^T * S^T * delta */
        /* Step 1a: temp1 = S^T * delta using KokkosBlas::gemv for better vectorization */
        KokkosBlas::SerialGemv<KokkosBlas::Trans::Transpose, KokkosBlas::Algo::Gemv::Unblocked>::invoke(1.0, S_i, delta_i, 0.0, temp1);

        /* Step 1b: temp2 = V^T * temp1 using KokkosBlas::gemv for better vectorization */
        KokkosBlas::SerialGemv<KokkosBlas::Trans::Transpose, KokkosBlas::Algo::Gemv::Unblocked>::invoke(1.0, V_i, temp1, 0.0, temp2);

        /* Step 1c: temp2 = temp2 / Lambda; floor Lambda by LETKF_EIGEN_EPS (see header) */
        for (int j = 0; j < m; j++) temp2(j) /= (Lambda_i(j) + LETKF_EIGEN_EPS);

        /* Step 1d: w = V * temp2 using KokkosBlas::gemv for better vectorization */
        KokkosBlas::SerialGemv<KokkosBlas::Trans::NoTranspose, KokkosBlas::Algo::Gemv::Unblocked>::invoke(1.0, V_i, temp2, 0.0, w_i);

        /* 2. Precompute 1/sqrt(Lambda) for ensemble update; same LETKF_EIGEN_EPS floor as above */
        for (int p = 0; p < m; p++) inv_sqrt_lambda_i(p) = 1.0 / Kokkos::sqrt(PetscRealPart(Lambda_i(p)) + LETKF_EIGEN_EPS);
      });
    Kokkos::fence();

    /* ===================================================================== */
    /* Step 3.1.3: Fused G computation and ensemble update                  */
    /* ===================================================================== */
    /* Compute E[i,:] = mean[i] + X[i,:] * G_i on-the-fly */
    /* G_i is computed column-by-column and immediately applied */
    /* This eliminates the need to store G_batch, saving m*m*n_batch memory */
    Kokkos::parallel_for(
      "FusedGComputeAndEnsembleUpdate", Kokkos::RangePolicy<exec_space>(0, n_batch_current), KOKKOS_LAMBDA(const int i_local) {
        PetscInt i_global = chunk_start + i_local;

        auto X_i    = Kokkos::subview(X_view, Kokkos::make_pair(i_global * ndof, (i_global + 1) * ndof), Kokkos::ALL());
        auto E_i    = Kokkos::subview(E_view, Kokkos::make_pair(i_global * ndof, (i_global + 1) * ndof), Kokkos::ALL());
        auto mean_i = Kokkos::subview(mean_view, Kokkos::make_pair(i_global * ndof, (i_global + 1) * ndof));

        auto V_i               = Kokkos::subview(V_batch, i_local, Kokkos::ALL(), Kokkos::ALL());
        auto w_i               = Kokkos::subview(w_batch, i_local, Kokkos::ALL());
        auto inv_sqrt_lambda_i = Kokkos::subview(inv_sqrt_lambda_batch, i_local, Kokkos::ALL());
        auto T_sqrt_i          = Kokkos::subview(T_sqrt_batch, i_local, Kokkos::ALL(), Kokkos::ALL());

        /* Initialize E_i with mean */
        for (int row = 0; row < ndof; row++) {
          PetscScalar m_val = mean_i(row);
          for (int col = 0; col < m; col++) E_i(row, col) = m_val;
        }

        /* Compute T_sqrt = V * diag(1/sqrt(Lambda)) * V^T */
        /* Optimized: Exploit symmetry - only compute upper triangle, then copy to lower */
        /* T_sqrt(j,k) = sum_p V(j,p) * V(k,p) / sqrt(Lambda(p)) for j <= k */
        for (int j = 0; j < m; j++) {
          for (int k = j; k < m; k++) {
            PetscScalar sum = 0.0;
            for (int p = 0; p < m; p++) sum += V_i(j, p) * V_i(k, p) * inv_sqrt_lambda_i(p);
            T_sqrt_i(j, k) = sum;
          }
        }
        /* Copy upper triangle to lower triangle (T_sqrt is symmetric) */
        for (int j = 0; j < m; j++) {
          for (int k = 0; k < j; k++) T_sqrt_i(j, k) = T_sqrt_i(k, j);
        }

        /* Compute E_i += X_i * G_i column-by-column */
        /* G_i(:,k) = w_i + sqrt(m-1) * T_sqrt_i(:,k) */
        for (int k = 0; k < m; k++) {
          /* Compute column k of G on-the-fly */
          for (int row = 0; row < ndof; row++) {
            PetscScalar sum = 0.0;
            for (int j = 0; j < m; j++) {
              /* G_i(j,k) = w_i(j) + sqrt(m-1) * T_sqrt_i(j,k) */
              PetscScalar G_jk = w_i(j) + sqrt_m_minus_1 * T_sqrt_i(j, k);
              sum += X_i(row, j) * G_jk;
            }
            E_i(row, k) += sum;
          }
        }
      });
    Kokkos::fence();
  }

  /* Cleanup workspace */
  /* NOTE: Workspace is now persistent in impl->eigen_work and impl->solver_handle */
  /* It will be destroyed in PetscDALETKFDestroyLocalization_Kokkos */

  /* Copy back updated ensemble if needed */
  if (e_is_copy) {
    Kokkos::View<PetscScalar **, Kokkos::LayoutLeft, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> dst(e_array, lda_e, m);
    PetscCallCXX(Kokkos::deep_copy(dst, e_managed));
  }

  /* Restore arrays */
  PetscCall(MatDenseRestoreArrayWriteAndMemType(en->ensemble, &e_array));
  PetscCall(VecRestoreArrayReadAndMemType(impl->mean, &mean_array));
  PetscCall(MatDenseRestoreArrayReadAndMemType(X, &x_array));

  /* Restore global observation arrays */
  PetscCall(VecRestoreArrayReadAndMemType(r_inv_sqrt_global, &r_inv_sqrt_global_array));
  PetscCall(VecRestoreArrayReadAndMemType(y_mean_global, &y_mean_global_array));
  PetscCall(VecRestoreArrayReadAndMemType(observation, &y_global_array));
  PetscCall(MatDenseRestoreArrayReadAndMemType(Z_global, &z_global_array));

  /* impl->n_nnz_local was populated by PetscDALETKFInstallQ() and is valid for the lifetime of Q
     (which only changes via InstallQ). Reading from impl avoids a redundant MatGetInfo here (and
     the AIJKOKKOS device->host sync it would trigger). */
  n_obs_total = (PetscReal)impl->n_nnz_local;
  flops       = 0.0;

  /* Step 2.1.2: Fused observation extraction and S/Delta computation */
  flops += n_obs_total * (2.0 + 2.0 * m);

  /* Step 2.1.4: Optimized T matrix formation */
  flops += n_obs_total * m * (m + 1);

  /* Step 3.1.2: Precompute w and inv_sqrt_lambda */
  flops += n_obs_total * 2.0 * m + (PetscReal)n_vertices * (4.0 * m * m + 3.0 * m);

  /* Step 3.1.3: Fused G computation and ensemble update */
  /* T_sqrt: 1.5*m^3 + 1.5*m^2 */
  flops += (PetscReal)n_vertices * (1.5 * m * m * m + 1.5 * m * m);
  /* E update: ndof * m * (4*m + 1) */
  /* Note: G_jk computation (2 flops) is inside the inner loop, so it's 2*m*ndof*m */
  /* Matrix product X*G (2 flops) is also 2*m*ndof*m */
  flops += (PetscReal)n_vertices * ndof * m * (4.0 * m + 1.0);

  PetscCall(PetscLogGpuFlops(flops));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* ========================================================================== */
/*                LETKF Global Analysis (LOC_NONE, Kokkos path)               */
/* ========================================================================== */

/*
  PetscDALETKFGlobalAnalysis_Kokkos - LOC_NONE LETKF analysis using device gemm/gemv.

  Mirrors the CPU LOC_NONE block in PetscDAEnsembleAnalysis_LETKF: device-side gemm for
  S^T*S, gemv for S^T*delta, and gemm for X*G; the m x m factor (T = (1/rho)I + S^T*S)
  is eigendecomposed on the host on every rank since m is small (ensemble size).
*/
PETSC_INTERN PetscErrorCode PetscDALETKFGlobalAnalysis_Kokkos(PetscDA da, PetscDA_LETKF *impl, PetscInt m, Mat X, Vec observation)
{
  using exec_space       = Kokkos::DefaultExecutionSpace;
  using view_2d          = Kokkos::View<PetscScalar **, Kokkos::LayoutLeft, exec_space>;
  using view_1d          = Kokkos::View<PetscScalar *, Kokkos::LayoutLeft, exec_space>;
  using view_2d_const_um = Kokkos::View<const PetscScalar **, Kokkos::LayoutLeft, exec_space, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
  using view_1d_const_um = Kokkos::View<const PetscScalar *, Kokkos::LayoutLeft, exec_space, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
  using view_2d_um       = Kokkos::View<PetscScalar **, Kokkos::LayoutLeft, exec_space, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
  using h_2d_const_um    = Kokkos::View<const PetscScalar **, Kokkos::LayoutLeft, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
  using h_1d_const_um    = Kokkos::View<const PetscScalar *, Kokkos::LayoutLeft, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
  using h_2d_um          = Kokkos::View<PetscScalar **, Kokkos::LayoutLeft, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
  using h_1d_um          = Kokkos::View<PetscScalar *, Kokkos::LayoutLeft, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
  MPI_Comm           comm;
  PetscReal          scale, sqrt_m_minus_1;
  PetscInt           n_obs_local, n_local_ens, s_lda, x_lda, e_lda, g_lda;
  PetscMemType       s_mt, d_mt, mean_mt, x_mt, e_mt;
  const PetscScalar *s_arr, *d_arr, *mean_arr, *x_arr, *g_host;
  const PetscScalar *s_dev, *d_dev, *mean_dev, *x_dev;
  PetscScalar       *e_arr, *e_dev, *gram_host, *sd_buf;
  PetscMPIInt        mMPI, mmMPI;
  PetscBool          e_is_copy = PETSC_FALSE;
  view_2d            S_managed, X_managed, E_managed, gram_dev, G_dev, XG_dev;
  view_1d            d_managed, mean_managed, Sd_dev;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)da, &comm));
  PetscCall(PetscKokkosInitializeCheck());

  scale          = 1.0 / PetscSqrtReal((PetscReal)(m - 1));
  sqrt_m_minus_1 = PetscSqrtReal((PetscReal)(m - 1));

  PetscCall(PetscDALETKFEnsureGlobalScratch(impl, m));

  /* S = R^{-1/2} * (Z - y_mean*1') / sqrt(m-1); delta_scaled = R^{-1/2} * (y - y_mean) */
  PetscCall(PetscDAEnsembleComputeNormalizedInnovationMatrix(impl->Z, impl->y_mean, impl->r_inv_sqrt, m, scale, impl->S));
  PetscCall(VecWAXPY(impl->delta_scaled, -1.0, impl->y_mean, observation));
  PetscCall(VecPointwiseMult(impl->delta_scaled, impl->delta_scaled, impl->r_inv_sqrt));

  PetscCall(MatGetLocalSize(impl->S, &n_obs_local, NULL));
  PetscCall(MatGetLocalSize(impl->en.ensemble, &n_local_ens, NULL));

  PetscCall(MatDenseGetArrayReadAndMemType(impl->S, &s_arr, &s_mt));
  PetscCall(MatDenseGetLDA(impl->S, &s_lda));
  PetscCall(VecGetArrayReadAndMemType(impl->delta_scaled, &d_arr, &d_mt));
  PetscCall(VecGetArrayReadAndMemType(impl->mean, &mean_arr, &mean_mt));
  PetscCall(MatDenseGetArrayReadAndMemType(X, &x_arr, &x_mt));
  PetscCall(MatDenseGetLDA(X, &x_lda));
  PetscCall(MatDenseGetArrayWriteAndMemType(impl->en.ensemble, &e_arr, &e_mt));
  PetscCall(MatDenseGetLDA(impl->en.ensemble, &e_lda));

  /* Mirror host arrays to device when needed. */
  s_dev    = s_arr;
  d_dev    = d_arr;
  mean_dev = mean_arr;
  x_dev    = x_arr;
  e_dev    = e_arr;

  if (s_mt == PETSC_MEMTYPE_HOST && n_obs_local > 0) {
    PetscCallCXX(S_managed = view_2d("S_managed", s_lda, m));
    PetscCallCXX(Kokkos::deep_copy(S_managed, h_2d_const_um(s_arr, s_lda, m)));
    s_dev = S_managed.data();
  }
  if (d_mt == PETSC_MEMTYPE_HOST && n_obs_local > 0) {
    PetscCallCXX(d_managed = view_1d("d_managed", n_obs_local));
    PetscCallCXX(Kokkos::deep_copy(d_managed, h_1d_const_um(d_arr, n_obs_local)));
    d_dev = d_managed.data();
  }
  if (mean_mt == PETSC_MEMTYPE_HOST && n_local_ens > 0) {
    PetscCallCXX(mean_managed = view_1d("mean_managed", n_local_ens));
    PetscCallCXX(Kokkos::deep_copy(mean_managed, h_1d_const_um(mean_arr, n_local_ens)));
    mean_dev = mean_managed.data();
  }
  /* X_managed / E_managed are only consumed inside the n_local_ens > 0 block below; allocating
     the mirrors on a rank with no ensemble rows would size to a non-positive (x_lda, e_lda) and
     waste a Kokkos View on data we never read or write. */
  if (x_mt == PETSC_MEMTYPE_HOST && n_local_ens > 0) {
    PetscCallCXX(X_managed = view_2d("X_managed", x_lda, m));
    PetscCallCXX(Kokkos::deep_copy(X_managed, h_2d_const_um(x_arr, x_lda, m)));
    x_dev = X_managed.data();
  }
  if (e_mt == PETSC_MEMTYPE_HOST && n_local_ens > 0) {
    PetscCallCXX(E_managed = view_2d("E_managed", e_lda, m));
    e_dev     = E_managed.data();
    e_is_copy = PETSC_TRUE;
  }

  /* Device gemm: gram = S^T * S over the active local rows [0, n_obs_local). */
  PetscCallCXX(gram_dev = view_2d("gram_dev", m, m));
  if (n_obs_local > 0) {
    view_2d_const_um S_full(s_dev, s_lda, m);
    auto             S_active = Kokkos::subview(S_full, Kokkos::make_pair((PetscInt)0, n_obs_local), Kokkos::ALL());
    KokkosBlas::gemm("T", "N", (PetscScalar)1.0, S_active, S_active, (PetscScalar)0.0, gram_dev);
  }
  Kokkos::fence();

  /* Mirror gram to host, allreduce, and feed the shared SELF-gram factorizer. PetscCalloc1
     so an n_obs_local == 0 rank (where the device gemm above is skipped, leaving gram_dev
     at Kokkos's default-zero state but a future refactor might also skip the deep_copy)
     contributes zeros to the allreduce instead of uninitialized bytes. */
  PetscCall(PetscCalloc1((size_t)m * m, &gram_host));
  PetscCallCXX(Kokkos::deep_copy(h_2d_um(gram_host, m, m), gram_dev));
  PetscCall(PetscMPIIntCast((PetscInt64)m * m, &mmMPI));
  PetscCallMPI(MPIU_Allreduce(MPI_IN_PLACE, gram_host, mmMPI, MPIU_SCALAR, MPIU_SUM, comm));
  PetscCall(PetscDAEnsembleTFactorFromGram(da, m, gram_host));
  PetscCall(PetscFree(gram_host));

  /* Device gemv: Sd = S^T * delta_scaled, then mirror + Allreduce. */
  PetscCallCXX(Sd_dev = view_1d("Sd_dev", m));
  if (n_obs_local > 0) {
    view_2d_const_um S_full(s_dev, s_lda, m);
    view_1d_const_um d_full(d_dev, n_obs_local);
    auto             S_active = Kokkos::subview(S_full, Kokkos::make_pair((PetscInt)0, n_obs_local), Kokkos::ALL());
    KokkosBlas::gemv("T", (PetscScalar)1.0, S_active, d_full, (PetscScalar)0.0, Sd_dev);
  }
  Kokkos::fence();

  /* Stage the device-side Sd into the persistent impl->s_transpose_delta scratch (matches the
     CPU path) and allreduce in place. */
  PetscCall(VecGetArray(impl->s_transpose_delta, &sd_buf));
  PetscCallCXX(Kokkos::deep_copy(h_1d_um(sd_buf, m), Sd_dev));
  PetscCall(PetscMPIIntCast(m, &mMPI));
  PetscCallMPI(MPIU_Allreduce(MPI_IN_PLACE, sd_buf, mMPI, MPIU_SCALAR, MPIU_SUM, comm));
  PetscCall(VecRestoreArray(impl->s_transpose_delta, &sd_buf));

  /* Restore S/delta read arrays before host-side T-inverse. */
  PetscCall(VecRestoreArrayReadAndMemType(impl->delta_scaled, &d_arr));
  PetscCall(MatDenseRestoreArrayReadAndMemType(impl->S, &s_arr));

  /* w = T^{-1} * (S^T * delta), all on PETSC_COMM_SELF. */
  PetscCall(PetscDAEnsembleApplyTInverse(da, impl->s_transpose_delta, impl->w));

  /* T_sqrt = T^{-1/2} on PETSC_COMM_SELF. */
  PetscCall(PetscDAEnsembleApplySqrtTInverse(da, NULL, impl->T_sqrt));

  /* G = w*1' + sqrt(m-1) * T_sqrt, m x m on PETSC_COMM_SELF in impl->w_ones. */
  PetscCall(PetscDALETKFReplicateWeightVector(impl->w, m, impl->w_ones));
  PetscCall(MatAXPY(impl->w_ones, sqrt_m_minus_1, impl->T_sqrt, SAME_NONZERO_PATTERN));

  /* Push G to device for the X*G gemm. impl->w_ones is a SELF SeqDense; LDA == m. */
  PetscCallCXX(G_dev = view_2d("G_dev", m, m));
  PetscCall(MatDenseGetArrayRead(impl->w_ones, &g_host));
  PetscCall(MatDenseGetLDA(impl->w_ones, &g_lda));
  PetscCheck(g_lda == m, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Unexpected LDA %" PetscInt_FMT " for SELF SeqDense w_ones (m=%" PetscInt_FMT ")", g_lda, m);
  PetscCallCXX(Kokkos::deep_copy(G_dev, h_2d_const_um(g_host, m, m)));
  PetscCall(MatDenseRestoreArrayRead(impl->w_ones, &g_host));

  /* Device gemm: XG = X_local * G, then E = mean*1' + XG. Allocate XG_dev only when
     this rank actually owns ensemble columns; consumers below are gated identically. */
  if (n_local_ens > 0) {
    view_2d_const_um X_full(x_dev, x_lda, m);
    auto             X_active = Kokkos::subview(X_full, Kokkos::make_pair((PetscInt)0, n_local_ens), Kokkos::ALL());
    view_2d_um       E_full(e_dev, e_lda, m);
    view_1d_const_um mean_full(mean_dev, n_local_ens);
    PetscInt         m_local = m;

    PetscCallCXX(XG_dev = view_2d("XG_dev", n_local_ens, m));
    KokkosBlas::gemm("N", "N", (PetscScalar)1.0, X_active, G_dev, (PetscScalar)0.0, XG_dev);
    Kokkos::parallel_for(
      "EnsembleUpdate_LOC_NONE", Kokkos::RangePolicy<exec_space>(0, n_local_ens), KOKKOS_LAMBDA(const int i) {
        PetscScalar mi = mean_full(i);
        for (PetscInt j = 0; j < m_local; j++) E_full(i, j) = mi + XG_dev(i, j);
      });
    Kokkos::fence();
  }

  PetscCall(VecRestoreArrayReadAndMemType(impl->mean, &mean_arr));
  PetscCall(MatDenseRestoreArrayReadAndMemType(X, &x_arr));

  if (e_is_copy && n_local_ens > 0) {
    /* Copy only the active rows [0, n_local_ens). E_managed is allocated to (e_lda, m) so the
       device-side strides line up with the host MatDense LDA, but the LDA-padding rows
       [n_local_ens, e_lda) are not touched by the analysis and must not be written back into
       the host buffer's opaque padding bytes. */
    Kokkos::View<PetscScalar **, Kokkos::LayoutLeft, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> dst(e_arr, e_lda, m);
    PetscCallCXX(Kokkos::deep_copy(Kokkos::subview(dst, Kokkos::make_pair((PetscInt)0, n_local_ens), Kokkos::ALL()), Kokkos::subview(E_managed, Kokkos::make_pair((PetscInt)0, n_local_ens), Kokkos::ALL())));
  }
  PetscCall(MatDenseRestoreArrayWriteAndMemType(impl->en.ensemble, &e_arr));
  PetscFunctionReturn(PETSC_SUCCESS);
}

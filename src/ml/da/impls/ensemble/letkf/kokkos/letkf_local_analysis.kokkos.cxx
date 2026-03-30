#include "../src/ml/da/impls/ensemble/letkf/letkf.h"
#include <petscblaslapack.h>
#include <Kokkos_Core.hpp>
#include <KokkosBlas.hpp>
#include <KokkosBatched_SVD_Decl.hpp>
#include <KokkosBatched_SVD_Serial_Impl.hpp>
#include <KokkosBatched_Gemm_Decl.hpp>
#include <KokkosBatched_Gemm_Serial_Impl.hpp>
#include <KokkosBatched_Util.hpp>

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
  #include <petscdevice_sycl.h>
#endif

/* ========================================================================== */
/*                    Batched Eigendecomposition for LETKF                    */
/* ========================================================================== */

/* Structure to hold reusable workspace for eigensolvers */
struct EigenWorkspace {
  /* Tracking for reuse */
  PetscInt max_chunk_size;
  PetscInt m;
  PetscInt n_obs_vertex;

  /* Persistent Kokkos Views */
  using exec_space = Kokkos::DefaultExecutionSpace;
  using view_3d    = Kokkos::View<PetscScalar ***, Kokkos::LayoutLeft, exec_space>;
  using view_2d    = Kokkos::View<PetscScalar **, Kokkos::LayoutLeft, exec_space>;

  view_3d Z_batch;
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

  EigenWorkspace() : max_chunk_size(0), m(0), n_obs_vertex(0), all_v(nullptr), all_lambda(nullptr), all_work(nullptr)
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
static PetscErrorCode BatchedEigenSolve_Host(Kokkos::View<PetscScalar ***, Kokkos::LayoutLeft, Kokkos::DefaultExecutionSpace> T_batch, Kokkos::View<PetscScalar **, Kokkos::LayoutLeft, Kokkos::DefaultExecutionSpace> Lambda_batch, Kokkos::View<PetscScalar ***, Kokkos::LayoutLeft, Kokkos::DefaultExecutionSpace> V_batch, PetscInt n_batch, PetscInt n_size, EigenWorkspace *work)
{
  PetscFunctionBegin;
  /* Create host mirrors and copy data in one operation */
  /* This is required for HIP+complex where create_mirror_view + deep_copy fails */
  auto T_host      = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), T_batch);
  auto Lambda_host = Kokkos::create_mirror_view(Kokkos::HostSpace(), Lambda_batch);
  auto V_host      = Kokkos::create_mirror_view(Kokkos::HostSpace(), V_batch);

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

      /* Copy T_host(i, :, :) to v_ptr (column-major) */
      for (PetscInt j = 0; j < n_size; j++) {
        for (PetscInt k = 0; k < n_size; k++) v_ptr[k + j * n_size] = T_host(i, k, j);
      }

    /* Compute eigendecomposition: T = V * Lambda * V^T */
  #if defined(PETSC_USE_COMPLEX)
      LAPACKsyev_("V", "U", &n, v_ptr, &lda, lambda_ptr, work_ptr, &lw, rwork_ptr, &info);
  #else
      LAPACKsyev_("V", "U", &n, v_ptr, &lda, lambda_ptr, work_ptr, &lw, &info);
  #endif

      if (info != 0) {
        /* We cannot return error code from lambda, so we just abort or ignore.
           In production code, we should use a reduction to report errors. */
        Kokkos::abort("LAPACK eigendecomposition failed in parallel region");
      }

      /* Copy results back to host views */
      for (PetscInt j = 0; j < n_size; j++) {
        Lambda_host(i, j) = (PetscScalar)lambda_ptr[j];
        for (PetscInt k = 0; k < n_size; k++) V_host(i, k, j) = v_ptr[k + j * n_size];
      }
    });

  /* Copy results back to device */
  Kokkos::deep_copy(Lambda_batch, Lambda_host);
  Kokkos::deep_copy(V_batch, V_host);
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
static PetscErrorCode BatchedEigenSolve_Device(Kokkos::View<PetscScalar ***, Kokkos::LayoutLeft, Kokkos::DefaultExecutionSpace> T_batch, Kokkos::View<PetscScalar **, Kokkos::LayoutLeft, Kokkos::DefaultExecutionSpace> Lambda_batch, Kokkos::View<PetscScalar ***, Kokkos::LayoutLeft, Kokkos::DefaultExecutionSpace> V_batch, PetscInt n_batch, PetscInt n_size, cusolverDnHandle_t cusolverH, EigenWorkspace *work)
{
  cusolverStatus_t cusolver_status;

  PetscFunctionBegin;
  /* Use pre-allocated workspace */
  syevjInfo_t  syevj_params = work->syevj_params;
  PetscScalar *d_work       = work->d_work;
  int         *d_info       = work->d_info;
  PetscScalar *d_A_contig   = work->d_A_contig;
  PetscScalar *d_W_contig   = work->d_W_contig;
  int          lwork        = work->lwork_device;

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
  int *h_info;
  PetscCall(PetscMalloc1(n_batch, &h_info));
  PetscCallCUDA(cudaMemcpy(h_info, d_info, sizeof(int) * n_batch, cudaMemcpyDeviceToHost));
  for (int i = 0; i < n_batch; i++) {
    if (h_info[i] != 0) PetscCheck(h_info[i] == 0, PETSC_COMM_SELF, PETSC_ERR_LIB, "cuSOLVER eigendecomposition failed for matrix %" PetscInt_FMT ": info=%d", i, h_info[i]);
  }
  PetscCall(PetscFree(h_info));

  /* Copy results back from contiguous layout to V_batch */
  Kokkos::parallel_for(
    "CopyResultsBack", Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, n_batch), KOKKOS_LAMBDA(const int i) {
      for (int j = 0; j < n_size; j++) {
        Lambda_batch(i, j) = d_W_contig[i * n_size + j];
        for (int k = 0; k < n_size; k++) V_batch(i, j, k) = d_A_contig[i * n_size * n_size + k * n_size + j];
      }
    });
  Kokkos::fence();
  PetscFunctionReturn(PETSC_SUCCESS);
}
  #elif defined(KOKKOS_ENABLE_HIP)
static PetscErrorCode BatchedEigenSolve_Device(Kokkos::View<PetscScalar ***, Kokkos::LayoutLeft, Kokkos::DefaultExecutionSpace> T_batch, Kokkos::View<PetscScalar **, Kokkos::LayoutLeft, Kokkos::DefaultExecutionSpace> Lambda_batch, Kokkos::View<PetscScalar ***, Kokkos::LayoutLeft, Kokkos::DefaultExecutionSpace> V_batch, PetscInt n_batch, PetscInt n_size, rocblas_handle rocblasH, EigenWorkspace *work)
{
  PetscFunctionBegin;
  /* Use pre-allocated workspace */
  PetscScalar *d_work = work->d_work;
  (void)d_work;
  int         *d_info     = work->d_info;
  PetscScalar *d_A_contig = work->d_A_contig;
  PetscScalar *d_W_contig = work->d_W_contig;

  /* Copy T_batch to contiguous layout for rocSOLVER */
  Kokkos::parallel_for(
    "ReorganizeForRocSOLVER", Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, n_batch), KOKKOS_LAMBDA(const int i) {
      for (int j = 0; j < n_size; j++) {
        for (int k = 0; k < n_size; k++) d_A_contig[i * n_size * n_size + k * n_size + j] = T_batch(i, j, k);
      }
    });
  Kokkos::fence();

    /* rocSOLVER doesn't have a native batched syevj, so we loop over batch */
    /* Use rocsolver_dsyevd which is more efficient than calling syev in a loop */
    #if defined(PETSC_USE_COMPLEX)
  SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Complex numbers not supported on HIP backend for LETKF");
    #else
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
    PetscCheck(hip_status == rocblas_status_success, PETSC_COMM_SELF, PETSC_ERR_LIB, "rocsolver_*syevd failed for batch %" PetscInt_FMT, i);
  }
    #endif

  /* Check info */
  int *h_info;
  PetscCall(PetscMalloc1(n_batch, &h_info));
  PetscCallHIP(hipMemcpy(h_info, d_info, sizeof(int) * n_batch, hipMemcpyDeviceToHost));
  for (int i = 0; i < n_batch; i++) {
    if (h_info[i] != 0) PetscCheck(h_info[i] == 0, PETSC_COMM_SELF, PETSC_ERR_LIB, "rocSOLVER eigendecomposition failed for matrix %" PetscInt_FMT ": info=%d", i, h_info[i]);
  }
  PetscCall(PetscFree(h_info));

  /* Copy results back from contiguous layout to V_batch */
  Kokkos::parallel_for(
    "CopyResultsBack", Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, n_batch), KOKKOS_LAMBDA(const int i) {
      for (int j = 0; j < n_size; j++) {
        Lambda_batch(i, j) = d_W_contig[i * n_size + j];
        for (int k = 0; k < n_size; k++) V_batch(i, j, k) = d_A_contig[i * n_size * n_size + k * n_size + j];
      }
    });
  Kokkos::fence();
  PetscFunctionReturn(PETSC_SUCCESS);
}
  #elif defined(KOKKOS_ENABLE_SYCL)
static PetscErrorCode BatchedEigenSolve_Device(Kokkos::View<PetscScalar ***, Kokkos::LayoutLeft, Kokkos::DefaultExecutionSpace> T_batch, Kokkos::View<PetscScalar **, Kokkos::LayoutLeft, Kokkos::DefaultExecutionSpace> Lambda_batch, Kokkos::View<PetscScalar ***, Kokkos::LayoutLeft, Kokkos::DefaultExecutionSpace> V_batch, PetscInt n_batch, PetscInt n_size, sycl::queue *q, EigenWorkspace *work)
{
  PetscFunctionBegin;
  /* Use pre-allocated workspace */
  PetscScalar *d_work     = work->d_work;
  int         *d_info     = work->d_info;
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

  /* oneMKL doesn't have a native batched syevd, so we loop over batch */
  /* Use oneapi::mkl::lapack::syevd which computes eigenvalues and eigenvectors */
  for (int i = 0; i < n_batch; i++) {
    PetscScalar *A_ptr    = d_A_contig + i * n_size * n_size;
    PetscScalar *W_ptr    = d_W_contig + i * n_size;
    int         *info_ptr = d_info + i;

    try {
    #if defined(PETSC_USE_REAL_SINGLE)
      oneapi::mkl::lapack::syevd(*q, oneapi::mkl::job::vec, oneapi::mkl::uplo::upper, n_size, A_ptr, n_size, W_ptr, d_work, work->lwork_device, info_ptr);
    #else
      oneapi::mkl::lapack::syevd(*q, oneapi::mkl::job::vec, oneapi::mkl::uplo::upper, n_size, A_ptr, n_size, W_ptr, d_work, work->lwork_device, info_ptr);
    #endif
      q->wait();
    } catch (sycl::exception const &e) {
      SETERRQ(PETSC_COMM_SELF, PETSC_ERR_LIB, "oneMKL syevd failed for batch %d: %s", i, e.what());
    }
  }

  /* Check info */
  int *h_info;
  PetscCall(PetscMalloc1(n_batch, &h_info));
  q->memcpy(h_info, d_info, sizeof(int) * n_batch).wait();
  for (int i = 0; i < n_batch; i++) {
    if (h_info[i] != 0) PetscCheck(h_info[i] == 0, PETSC_COMM_SELF, PETSC_ERR_LIB, "oneMKL eigendecomposition failed for matrix %" PetscInt_FMT ": info=%d", i, h_info[i]);
  }
  PetscCall(PetscFree(h_info));

  /* Copy results back from contiguous layout to V_batch */
  Kokkos::parallel_for(
    "CopyResultsBack", Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, n_batch), KOKKOS_LAMBDA(const int i) {
      for (int j = 0; j < n_size; j++) {
        Lambda_batch(i, j) = d_W_contig[i * n_size + j];
        for (int k = 0; k < n_size; k++) V_batch(i, j, k) = d_A_contig[i * n_size * n_size + k * n_size + j];
      }
    });
  Kokkos::fence();
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
static PetscErrorCode BatchedEigenSolve(Kokkos::View<PetscScalar ***, Kokkos::LayoutLeft, Kokkos::DefaultExecutionSpace> T_batch, Kokkos::View<PetscScalar **, Kokkos::LayoutLeft, Kokkos::DefaultExecutionSpace> Lambda_batch, Kokkos::View<PetscScalar ***, Kokkos::LayoutLeft, Kokkos::DefaultExecutionSpace> V_batch, PetscInt n_batch, PetscInt n_size, cusolverDnHandle_t device_handle, EigenWorkspace *work)
{
  PetscFunctionBegin;
  PetscCall(BatchedEigenSolve_Device(T_batch, Lambda_batch, V_batch, n_batch, n_size, device_handle, work));
  PetscFunctionReturn(PETSC_SUCCESS);
}
  #elif defined(KOKKOS_ENABLE_HIP)
static PetscErrorCode BatchedEigenSolve(Kokkos::View<PetscScalar ***, Kokkos::LayoutLeft, Kokkos::DefaultExecutionSpace> T_batch, Kokkos::View<PetscScalar **, Kokkos::LayoutLeft, Kokkos::DefaultExecutionSpace> Lambda_batch, Kokkos::View<PetscScalar ***, Kokkos::LayoutLeft, Kokkos::DefaultExecutionSpace> V_batch, PetscInt n_batch, PetscInt n_size, rocblas_handle device_handle, EigenWorkspace *work)
{
  PetscFunctionBegin;
  PetscCall(BatchedEigenSolve_Device(T_batch, Lambda_batch, V_batch, n_batch, n_size, device_handle, work));
  PetscFunctionReturn(PETSC_SUCCESS);
}
  #elif defined(KOKKOS_ENABLE_SYCL)
static PetscErrorCode BatchedEigenSolve(Kokkos::View<PetscScalar ***, Kokkos::LayoutLeft, Kokkos::DefaultExecutionSpace> T_batch, Kokkos::View<PetscScalar **, Kokkos::LayoutLeft, Kokkos::DefaultExecutionSpace> Lambda_batch, Kokkos::View<PetscScalar ***, Kokkos::LayoutLeft, Kokkos::DefaultExecutionSpace> V_batch, PetscInt n_batch, PetscInt n_size, sycl::queue *device_handle, EigenWorkspace *work)
{
  PetscFunctionBegin;
  PetscCall(BatchedEigenSolve_Device(T_batch, Lambda_batch, V_batch, n_batch, n_size, device_handle, work));
  PetscFunctionReturn(PETSC_SUCCESS);
}
  #endif
#else
static PetscErrorCode BatchedEigenSolve(Kokkos::View<PetscScalar ***, Kokkos::LayoutLeft, Kokkos::DefaultExecutionSpace> T_batch, Kokkos::View<PetscScalar **, Kokkos::LayoutLeft, Kokkos::DefaultExecutionSpace> Lambda_batch, Kokkos::View<PetscScalar ***, Kokkos::LayoutLeft, Kokkos::DefaultExecutionSpace> V_batch, PetscInt n_batch, PetscInt n_size, EigenWorkspace *work)
{
  PetscFunctionBegin;
  PetscCall(BatchedEigenSolve_Host(T_batch, Lambda_batch, V_batch, n_batch, n_size, work));
  PetscFunctionReturn(PETSC_SUCCESS);
}
#endif

/*
  PetscDALETKFSetupLocalization_Kokkos - Prepares device views for localization matrix Q
*/
PetscErrorCode PetscDALETKFSetupLocalization_Kokkos(PetscDA_LETKF *impl, Mat H)
{
  PetscInt nrows;

  PetscFunctionBegin;
  PetscCheck(impl->Q, PETSC_COMM_SELF, PETSC_ERR_LIB, "impl->Q = 0");
  PetscCall(PetscKokkosInitializeCheck());

  /* Get CSR data */
  PetscInt rstart, rend, i, nnz;
  PetscCall(MatGetOwnershipRange(impl->Q, &rstart, &rend));
  nrows = rend - rstart;

  /* Create IS for local observations needed by this process */
  /* We need to find all unique column indices in the local rows of Q */
  {
    PetscInt     *obs_indices;
    PetscInt      n_obs_local_total = 0;
    PetscInt      max_obs           = nrows * impl->n_obs_vertex;
    PetscInt      count             = 0;
    PetscHMapI    ht;
    PetscHashIter iter;
    PetscBool     missing;

    PetscCall(PetscHMapICreate(&ht));
    PetscCall(PetscMalloc1(max_obs, &obs_indices));

    for (i = 0; i < nrows; i++) {
      const PetscInt    *cols;
      const PetscScalar *vals;
      PetscCall(MatGetRow(impl->Q, rstart + i, &nnz, &cols, &vals));
      for (PetscInt k = 0; k < nnz; k++) {
        PetscCall(PetscHMapIPut(ht, cols[k], &iter, &missing));
        if (missing) {
          obs_indices[count] = cols[k];
          count++;
        }
      }
      PetscCall(MatRestoreRow(impl->Q, rstart + i, &nnz, &cols, &vals));
    }
    n_obs_local_total = count;

    /* Sort indices for consistent ordering */
    PetscCall(PetscSortInt(n_obs_local_total, obs_indices));

    /* Create IS and VecScatter */
    PetscCall(ISCreateGeneral(PETSC_COMM_SELF, n_obs_local_total, obs_indices, PETSC_COPY_VALUES, &impl->obs_is_local));

    /* Create global-to-local map for observations */
    PetscCall(PetscHMapICreate(&impl->obs_g2l));
    for (i = 0; i < n_obs_local_total; i++) {
      PetscCall(PetscHMapIPut(impl->obs_g2l, obs_indices[i], &iter, &missing));
      PetscCall(PetscHMapIIterSet(impl->obs_g2l, iter, i));
    }

    PetscCall(PetscFree(obs_indices));
    PetscCall(PetscHMapIDestroy(&ht));
  }

  /* Create work vectors and scatter context */
  {
    PetscInt n_obs_local_total;
    PetscCall(ISGetLocalSize(impl->obs_is_local, &n_obs_local_total));

    PetscCall(VecCreateSeq(PETSC_COMM_SELF, n_obs_local_total, &impl->obs_work));
    PetscCall(VecCreateSeq(PETSC_COMM_SELF, n_obs_local_total, &impl->y_mean_work));
    PetscCall(VecCreateSeq(PETSC_COMM_SELF, n_obs_local_total, &impl->r_inv_sqrt_work));

    Vec gvec;
    IS  is_to;
    PetscCall(MatCreateVecs(H, NULL, &gvec)); /* Create template global vector (left vector = rows = observations) */
    PetscCall(ISCreateStride(PETSC_COMM_SELF, n_obs_local_total, 0, 1, &is_to));
    PetscCall(VecScatterCreate(gvec, impl->obs_is_local, impl->obs_work, is_to, &impl->obs_scat));
    PetscCall(VecDestroy(&gvec));
    PetscCall(ISDestroy(&is_to));
  }

  /* Define View types */
  using view_1d_int    = Kokkos::View<PetscInt *, Kokkos::LayoutLeft>;
  using view_1d_scalar = Kokkos::View<PetscScalar *, Kokkos::LayoutLeft>;

  /* Allocate device views */
  view_1d_int    *d_Q_i = new view_1d_int("Q_i", nrows + 1);
  view_1d_int    *d_Q_j = new view_1d_int("Q_j", nrows * impl->n_obs_vertex);
  view_1d_scalar *d_Q_a = new view_1d_scalar("Q_a", nrows * impl->n_obs_vertex);

  /* Create host mirrors */
  auto h_Q_i = Kokkos::create_mirror_view(*d_Q_i);
  auto h_Q_j = Kokkos::create_mirror_view(*d_Q_j);
  auto h_Q_a = Kokkos::create_mirror_view(*d_Q_a);

  /* Fill host mirrors with LOCAL indices into obs_work */
  h_Q_i(0) = 0;
  for (i = 0; i < nrows; i++) {
    const PetscInt    *cols;
    const PetscScalar *vals;
    PetscCall(MatGetRow(impl->Q, rstart + i, &nnz, &cols, &vals));
    h_Q_i(i + 1) = h_Q_i(i) + nnz;
    for (PetscInt k = 0; k < nnz; k++) {
      PetscInt local_idx;
      PetscCall(ISLocate(impl->obs_is_local, cols[k], &local_idx));
      PetscCheck(local_idx >= 0, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Observation index %" PetscInt_FMT " not found in local IS", cols[k]);
      h_Q_j(h_Q_i(i) + k) = local_idx;
      h_Q_a(h_Q_i(i) + k) = vals[k];
    }
    PetscCall(MatRestoreRow(impl->Q, rstart + i, &nnz, &cols, &vals));
  }

  /* Copy to device */
  Kokkos::deep_copy(*d_Q_i, h_Q_i);
  Kokkos::deep_copy(*d_Q_j, h_Q_j);
  Kokkos::deep_copy(*d_Q_a, h_Q_a);

  /* Store in impl */
  PetscCheck(!impl->Q_device_i, PETSC_COMM_SELF, PETSC_ERR_LIB, "impl->Q = 0");
  impl->Q_device_i = static_cast<void *>(d_Q_i);
  impl->Q_device_j = static_cast<void *>(d_Q_j);
  impl->Q_device_a = static_cast<void *>(d_Q_a);
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PetscDALETKFDestroyLocalization_Kokkos(PetscDA_LETKF *impl)
{
  PetscFunctionBegin;
  PetscCall(VecDestroy(&impl->obs_work));
  PetscCall(VecDestroy(&impl->y_mean_work));
  PetscCall(VecDestroy(&impl->r_inv_sqrt_work));
  PetscCall(VecScatterDestroy(&impl->obs_scat));
  PetscCall(MatDestroy(&impl->Z_work));
  PetscCall(PetscHMapIDestroy(&impl->obs_g2l));
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

  /* Destroy solver handle and workspace */
  if (impl->eigen_work) {
    EigenWorkspace *work = static_cast<EigenWorkspace *>(impl->eigen_work);

#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP) || defined(KOKKOS_ENABLE_SYCL)
  #if defined(KOKKOS_ENABLE_CUDA)
    PetscCallCUDA(cudaFree(work->d_A_contig));
    PetscCallCUDA(cudaFree(work->d_W_contig));
    PetscCallCUDA(cudaFree(work->d_work));
    PetscCallCUDA(cudaFree(work->d_info));
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
  PetscDALETKFLocalAnalysis_GPU - Performs local LETKF analysis for all grid points (Kokkos version)

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
  This function performs the local analysis loop for LETKF, processing each grid point
  independently using its local observations defined by the localization matrix Q.
  This is the CPU version that does not use Kokkos acceleration.

  All local analysis workspace objects (Z_local, S_local, T_sqrt_local, G_local, y_local,
  y_mean_local, delta_scaled_local, r_inv_sqrt_local, w_local, s_transpose_delta) are
  created with PETSC_COMM_SELF because the analysis at each vertex is serial and independent.
*/
PetscErrorCode PetscDALETKFLocalAnalysis_GPU(PetscDA da, PetscDA_LETKF *impl, PetscInt m, PetscInt n_vertices, Mat X, Vec observation, Mat Z_global, Vec y_mean_global, Vec r_inv_sqrt_global)
{
  PetscDA_Ensemble *en = (PetscDA_Ensemble *)da->data;
  PetscInt          ndof;
  PetscReal         sqrt_m_minus_1, scale, inflation_inv;

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
  using exec_space = Kokkos::DefaultExecutionSpace;
  using view_3d    = Kokkos::View<PetscScalar ***, Kokkos::LayoutLeft, exec_space>;
  using view_2d    = Kokkos::View<PetscScalar **, Kokkos::LayoutLeft, exec_space>;

  /* ===================================================================== */
  /* Step 2.1.2a: Pre-extract Q matrix CSR data for device access        */
  /* ===================================================================== */
  using view_1d_int_const    = Kokkos::View<const PetscInt *, Kokkos::LayoutLeft, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
  using view_1d_scalar_const = Kokkos::View<const PetscScalar *, Kokkos::LayoutLeft, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
  using view_1d_int          = Kokkos::View<PetscInt *, Kokkos::LayoutLeft>;
  using view_1d_scalar       = Kokkos::View<PetscScalar *, Kokkos::LayoutLeft>;

  view_1d_int_const    Q_i_view;
  view_1d_int_const    Q_j_view;
  view_1d_scalar_const Q_a_view;

  if (impl->Q_device_i) {
    /* Use pre-allocated device views */
    view_1d_int    *d_Q_i = static_cast<view_1d_int *>(impl->Q_device_i);
    view_1d_int    *d_Q_j = static_cast<view_1d_int *>(impl->Q_device_j);
    view_1d_scalar *d_Q_a = static_cast<view_1d_scalar *>(impl->Q_device_a);

    Q_i_view = view_1d_int_const(d_Q_i->data(), d_Q_i->extent(0));
    Q_j_view = view_1d_int_const(d_Q_j->data(), d_Q_j->extent(0));
    Q_a_view = view_1d_scalar_const(d_Q_a->data(), d_Q_a->extent(0));
  } else {
    /* Fallback to host pointers (unsafe if not UVM) */
    PetscCheck(PETSC_FALSE, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Q matrix must be setup with PetscDALETKFSetupLocalization_Kokkos");
  }

  /* Get global observation data arrays */
  const PetscScalar *z_global_array, *y_global_array, *y_mean_global_array, *r_inv_sqrt_global_array;
  PetscInt           lda_z_global;
  PetscMemType       z_mem_type, y_mem_type, y_mean_mem_type, r_inv_sqrt_mem_type;

  PetscCall(MatDenseGetArrayReadAndMemType(Z_global, &z_global_array, &z_mem_type));
  PetscCall(VecGetArrayReadAndMemType(observation, &y_global_array, &y_mem_type));
  PetscCall(VecGetArrayReadAndMemType(y_mean_global, &y_mean_global_array, &y_mean_mem_type));
  PetscCall(VecGetArrayReadAndMemType(r_inv_sqrt_global, &r_inv_sqrt_global_array, &r_inv_sqrt_mem_type));
  PetscCall(MatDenseGetLDA(Z_global, &lda_z_global));

  /* Handle memory mirroring for observation data */
  Kokkos::View<PetscScalar **, Kokkos::LayoutLeft, exec_space> z_managed;
  Kokkos::View<PetscScalar *, Kokkos::LayoutLeft, exec_space>  y_managed;
  Kokkos::View<PetscScalar *, Kokkos::LayoutLeft, exec_space>  y_mean_managed;
  Kokkos::View<PetscScalar *, Kokkos::LayoutLeft, exec_space>  r_inv_sqrt_managed;

  const PetscScalar *z_ptr          = z_global_array;
  const PetscScalar *y_ptr          = y_global_array;
  const PetscScalar *y_mean_ptr     = y_mean_global_array;
  const PetscScalar *r_inv_sqrt_ptr = r_inv_sqrt_global_array;

  if (z_mem_type == PETSC_MEMTYPE_HOST) {
    z_managed = Kokkos::View<PetscScalar **, Kokkos::LayoutLeft, exec_space>("z_managed", lda_z_global, m);
    Kokkos::View<const PetscScalar **, Kokkos::LayoutLeft, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> src(z_global_array, lda_z_global, m);
    Kokkos::deep_copy(z_managed, src);
    z_ptr = z_managed.data();
  }
  if (y_mem_type == PETSC_MEMTYPE_HOST) {
    y_managed = Kokkos::View<PetscScalar *, Kokkos::LayoutLeft, exec_space>("y_managed", lda_z_global);
    Kokkos::View<const PetscScalar *, Kokkos::LayoutLeft, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> src(y_global_array, lda_z_global);
    Kokkos::deep_copy(y_managed, src);
    y_ptr = y_managed.data();
  }
  if (y_mean_mem_type == PETSC_MEMTYPE_HOST) {
    y_mean_managed = Kokkos::View<PetscScalar *, Kokkos::LayoutLeft, exec_space>("y_mean_managed", lda_z_global);
    Kokkos::View<const PetscScalar *, Kokkos::LayoutLeft, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> src(y_mean_global_array, lda_z_global);
    Kokkos::deep_copy(y_mean_managed, src);
    y_mean_ptr = y_mean_managed.data();
  }
  if (r_inv_sqrt_mem_type == PETSC_MEMTYPE_HOST) {
    r_inv_sqrt_managed = Kokkos::View<PetscScalar *, Kokkos::LayoutLeft, exec_space>("r_inv_sqrt_managed", lda_z_global);
    Kokkos::View<const PetscScalar *, Kokkos::LayoutLeft, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> src(r_inv_sqrt_global_array, lda_z_global);
    Kokkos::deep_copy(r_inv_sqrt_managed, src);
    r_inv_sqrt_ptr = r_inv_sqrt_managed.data();
  }

  /* Create unmanaged Kokkos views for global observation data */
  using view_2d_unmanaged = Kokkos::View<const PetscScalar **, Kokkos::LayoutLeft, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
  using view_1d_unmanaged = Kokkos::View<const PetscScalar *, Kokkos::LayoutLeft, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

  view_2d_unmanaged Z_global_view(z_ptr, lda_z_global, m);
  view_1d_unmanaged y_global_view(y_ptr, lda_z_global);
  view_1d_unmanaged y_mean_global_view(y_mean_ptr, lda_z_global);
  view_1d_unmanaged r_inv_sqrt_global_view(r_inv_sqrt_ptr, lda_z_global);

  /* Get access to global X matrix and mean vector */
  const PetscScalar *x_array, *mean_array;
  PetscScalar       *e_array;
  PetscInt           lda_x, lda_e;
  PetscMemType       x_mem_type, mean_mem_type, e_mem_type;

  PetscCall(MatDenseGetArrayReadAndMemType(X, &x_array, &x_mem_type));
  PetscCall(VecGetArrayReadAndMemType(impl->mean, &mean_array, &mean_mem_type));
  PetscCall(MatDenseGetArrayWriteAndMemType(en->ensemble, &e_array, &e_mem_type));
  PetscCall(MatDenseGetLDA(X, &lda_x));
  PetscCall(MatDenseGetLDA(en->ensemble, &lda_e));

  /* Handle memory mirroring for state data */
  Kokkos::View<PetscScalar **, Kokkos::LayoutLeft, exec_space> x_managed;
  Kokkos::View<PetscScalar *, Kokkos::LayoutLeft, exec_space>  mean_managed;
  Kokkos::View<PetscScalar **, Kokkos::LayoutLeft, exec_space> e_managed;

  const PetscScalar *x_ptr     = x_array;
  const PetscScalar *mean_ptr  = mean_array;
  PetscScalar       *e_ptr     = e_array;
  bool               e_is_copy = false;

  if (x_mem_type == PETSC_MEMTYPE_HOST) {
    x_managed = Kokkos::View<PetscScalar **, Kokkos::LayoutLeft, exec_space>("x_managed", lda_x, m);
    Kokkos::View<const PetscScalar **, Kokkos::LayoutLeft, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> src(x_array, lda_x, m);
    Kokkos::deep_copy(x_managed, src);
    x_ptr = x_managed.data();
  }
  if (mean_mem_type == PETSC_MEMTYPE_HOST) {
    mean_managed = Kokkos::View<PetscScalar *, Kokkos::LayoutLeft, exec_space>("mean_managed", lda_x);
    Kokkos::View<const PetscScalar *, Kokkos::LayoutLeft, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> src(mean_array, lda_x);
    Kokkos::deep_copy(mean_managed, src);
    mean_ptr = mean_managed.data();
  }
  if (e_mem_type == PETSC_MEMTYPE_HOST) {
    e_managed = Kokkos::View<PetscScalar **, Kokkos::LayoutLeft, exec_space>("e_managed", lda_e, m);
    Kokkos::View<PetscScalar **, Kokkos::LayoutLeft, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> src(e_array, lda_e, m);
    Kokkos::deep_copy(e_managed, src);
    e_ptr     = e_managed.data();
    e_is_copy = true;
  }

  /* Create unmanaged Kokkos views for global data */
  using view_2d_unmanaged_write = Kokkos::View<PetscScalar **, Kokkos::LayoutLeft, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
  view_2d_unmanaged       X_view(const_cast<PetscScalar *>(x_ptr), lda_x, m);
  view_1d_unmanaged       mean_view(mean_ptr, lda_x);
  view_2d_unmanaged_write E_view(e_ptr, lda_e, m);

  /* Determine chunk size to avoid OOM on large grids */
  PetscInt chunk_size;
  if (impl->batch_size > 0) {
    chunk_size = impl->batch_size;
  } else {
    /* Target ~2GB workspace. Approx memory per point: m*m*8 (T) + p*m*8 (Z) */
    /* With reuse: m*m*8 + p*m*8 */
    PetscInt mem_per_point = sizeof(PetscScalar) * (m * m + impl->n_obs_vertex * m);
    chunk_size             = (PetscInt)(2.0 * 1024 * 1024 * 1024 / mem_per_point);
    /* Clamp to reasonable max to avoid huge allocations even if memory allows */
    if (chunk_size > 32768) chunk_size = 32768;
  }

  if (chunk_size < 1) chunk_size = 1;
  if (chunk_size > n_vertices) chunk_size = n_vertices;

  /* OPTIMIZATION: Create device solver handle once, reuse across chunks */
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP) || defined(KOKKOS_ENABLE_SYCL)
  #if defined(KOKKOS_ENABLE_CUDA)
  cusolverDnHandle_t device_handle = nullptr;
  cusolverStatus_t   cusolver_status;
  if (impl->solver_handle) {
    device_handle = static_cast<cusolverDnHandle_t>(impl->solver_handle);
  } else {
    cusolver_status = cusolverDnCreate(&device_handle);
    PetscCheck(cusolver_status == CUSOLVER_STATUS_SUCCESS, PETSC_COMM_SELF, PETSC_ERR_LIB, "cusolverDnCreate failed");
    impl->solver_handle = static_cast<void *>(device_handle);
  }
  #elif defined(KOKKOS_ENABLE_HIP)
  rocblas_handle device_handle = nullptr;
  if (impl->solver_handle) {
    device_handle = static_cast<rocblas_handle>(impl->solver_handle);
  } else {
    rocblas_status hip_status = rocblas_create_handle(&device_handle);
    PetscCheck(hip_status == rocblas_status_success, PETSC_COMM_SELF, PETSC_ERR_LIB, "rocblas_create_handle failed");
    impl->solver_handle = static_cast<void *>(device_handle);
  }
  #elif defined(KOKKOS_ENABLE_SYCL)
  sycl::queue *device_handle = nullptr;
  if (impl->solver_handle) {
    device_handle = static_cast<sycl::queue *>(impl->solver_handle);
  } else {
    device_handle       = new sycl::queue(sycl::gpu_selector_v);
    impl->solver_handle = static_cast<void *>(device_handle);
  }
  #endif
#endif

  /* ===================================================================== */
  /* OPTIMIZATION: Hoist allocations outside the chunk loop                */
  /* ===================================================================== */
  /* Allocate Kokkos Views once for the maximum chunk size */
  PetscInt n_obs_vertex_copy = impl->n_obs_vertex;

  EigenWorkspace *eigen_work = static_cast<EigenWorkspace *>(impl->eigen_work);
  if (!eigen_work) {
    eigen_work       = new EigenWorkspace();
    impl->eigen_work = static_cast<void *>(eigen_work);
  }

  /* Check if reallocation is needed */
  if (eigen_work->max_chunk_size < chunk_size || eigen_work->m != m || eigen_work->n_obs_vertex != n_obs_vertex_copy) {
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
    eigen_work->n_obs_vertex   = n_obs_vertex_copy;

    /* Allocate Kokkos Views */
    eigen_work->Z_batch               = view_3d("Z_batch", chunk_size, n_obs_vertex_copy, m);
    eigen_work->S_batch               = eigen_work->Z_batch;
    eigen_work->T_batch               = view_3d("T_batch", chunk_size, m, m);
    eigen_work->V_batch               = eigen_work->T_batch;
    eigen_work->Lambda_batch          = view_2d("Lambda_batch", chunk_size, m);
    eigen_work->T_sqrt_batch          = view_3d("T_sqrt_batch", chunk_size, m, m);
    eigen_work->w_batch               = view_2d("w_batch", chunk_size, m);
    eigen_work->delta_batch           = view_2d("delta_batch", chunk_size, n_obs_vertex_copy);
    eigen_work->y_batch               = view_2d("y_batch", chunk_size, n_obs_vertex_copy);
    eigen_work->y_mean_batch          = view_2d("y_mean_batch", chunk_size, n_obs_vertex_copy);
    eigen_work->r_inv_sqrt_batch      = view_2d("r_inv_sqrt_batch", chunk_size, n_obs_vertex_copy);
    eigen_work->temp1_batch           = view_2d("temp1_batch", chunk_size, m);
    eigen_work->temp2_batch           = view_2d("temp2_batch", chunk_size, m);
    eigen_work->inv_sqrt_lambda_batch = view_2d("inv_sqrt_lambda_batch", chunk_size, m);

    /* Allocate solver workspace */
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP) || defined(KOKKOS_ENABLE_SYCL)
  #if defined(KOKKOS_ENABLE_CUDA)
    {
      /* Create syevj params */
      cusolver_status = cusolverDnCreateSyevjInfo(&eigen_work->syevj_params);
      PetscCheck(cusolver_status == CUSOLVER_STATUS_SUCCESS, PETSC_COMM_SELF, PETSC_ERR_LIB, "cusolverDnCreateSyevjInfo failed");

      /* Set default params */
      cusolverDnXsyevjSetTolerance(eigen_work->syevj_params, 1e-7);
      cusolverDnXsyevjSetMaxSweeps(eigen_work->syevj_params, 100);
      cusolverDnXsyevjSetSortEig(eigen_work->syevj_params, 1); /* Sort eigenvalues */

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
        /* rocsolver_dsyevd does not support size query via -1.
         We use a safe upper bound estimate based on LAPACK dsyevd requirements.
      */
    #if defined(PETSC_USE_COMPLEX)
      int lwork = 0; /* Complex not supported on device */
    #else
      int lwork = 1 + 6 * m + 2 * m * m;
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
      /* Query workspace size for oneapi::mkl::lapack::syevd */
      /* For syevd, workspace size is typically: */
      /* lwork >= 1 + 6*n + 2*n*n for real, or */
      /* lwork >= 2*n + n*n for complex */
      int lwork;
    #if defined(PETSC_USE_COMPLEX)
      lwork = 2 * m + m * m;
    #else
      lwork = 1 + 6 * m + 2 * m * m;
    #endif
      eigen_work->lwork_device = lwork;

      /* Allocate workspace using SYCL malloc_device */
      eigen_work->d_work     = sycl::malloc_device<PetscScalar>(lwork, *device_handle);
      eigen_work->d_info     = sycl::malloc_device<int>(chunk_size, *device_handle);
      eigen_work->d_A_contig = sycl::malloc_device<PetscScalar>(chunk_size * m * m, *device_handle);
      eigen_work->d_W_contig = sycl::malloc_device<PetscScalar>(chunk_size * m, *device_handle);
      PetscCheck(eigen_work->d_work && eigen_work->d_info && eigen_work->d_A_contig && eigen_work->d_W_contig, PETSC_COMM_SELF, PETSC_ERR_MEM, "SYCL memory allocation failed");
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
      PetscCheck(info == 0, PETSC_COMM_SELF, PETSC_ERR_LIB, "LAPACK workspace query failed");
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

  /* Create aliases for current function use */
  view_3d Z_batch_alloc               = eigen_work->Z_batch;
  view_3d S_batch_alloc               = eigen_work->S_batch;
  view_3d T_batch_alloc               = eigen_work->T_batch;
  view_3d V_batch_alloc               = eigen_work->V_batch;
  view_2d Lambda_batch_alloc          = eigen_work->Lambda_batch;
  view_3d T_sqrt_batch_alloc          = eigen_work->T_sqrt_batch;
  view_2d w_batch_alloc               = eigen_work->w_batch;
  view_2d delta_batch_alloc           = eigen_work->delta_batch;
  view_2d y_batch_alloc               = eigen_work->y_batch;
  view_2d y_mean_batch_alloc          = eigen_work->y_mean_batch;
  view_2d r_inv_sqrt_batch_alloc      = eigen_work->r_inv_sqrt_batch;
  view_2d temp1_batch_alloc           = eigen_work->temp1_batch;
  view_2d temp2_batch_alloc           = eigen_work->temp2_batch;
  view_2d inv_sqrt_lambda_batch_alloc = eigen_work->inv_sqrt_lambda_batch;

  /* Loop over chunks */
  for (PetscInt chunk_start = 0; chunk_start < n_vertices; chunk_start += chunk_size) {
    PetscInt chunk_end       = (chunk_start + chunk_size > n_vertices) ? n_vertices : chunk_start + chunk_size;
    PetscInt n_batch_current = chunk_end - chunk_start;

    /* Create subviews for current batch size */
    auto Z_batch               = Kokkos::subview(Z_batch_alloc, Kokkos::make_pair(0, (int)n_batch_current), Kokkos::ALL(), Kokkos::ALL());
    auto S_batch               = Kokkos::subview(S_batch_alloc, Kokkos::make_pair(0, (int)n_batch_current), Kokkos::ALL(), Kokkos::ALL());
    auto T_batch               = Kokkos::subview(T_batch_alloc, Kokkos::make_pair(0, (int)n_batch_current), Kokkos::ALL(), Kokkos::ALL());
    auto V_batch               = Kokkos::subview(V_batch_alloc, Kokkos::make_pair(0, (int)n_batch_current), Kokkos::ALL(), Kokkos::ALL());
    auto Lambda_batch          = Kokkos::subview(Lambda_batch_alloc, Kokkos::make_pair(0, (int)n_batch_current), Kokkos::ALL());
    auto T_sqrt_batch          = Kokkos::subview(T_sqrt_batch_alloc, Kokkos::make_pair(0, (int)n_batch_current), Kokkos::ALL(), Kokkos::ALL());
    auto w_batch               = Kokkos::subview(w_batch_alloc, Kokkos::make_pair(0, (int)n_batch_current), Kokkos::ALL());
    auto delta_batch           = Kokkos::subview(delta_batch_alloc, Kokkos::make_pair(0, (int)n_batch_current), Kokkos::ALL());
    auto y_batch               = Kokkos::subview(y_batch_alloc, Kokkos::make_pair(0, (int)n_batch_current), Kokkos::ALL());
    auto y_mean_batch          = Kokkos::subview(y_mean_batch_alloc, Kokkos::make_pair(0, (int)n_batch_current), Kokkos::ALL());
    auto r_inv_sqrt_batch      = Kokkos::subview(r_inv_sqrt_batch_alloc, Kokkos::make_pair(0, (int)n_batch_current), Kokkos::ALL());
    auto temp1_batch           = Kokkos::subview(temp1_batch_alloc, Kokkos::make_pair(0, (int)n_batch_current), Kokkos::ALL());
    auto temp2_batch           = Kokkos::subview(temp2_batch_alloc, Kokkos::make_pair(0, (int)n_batch_current), Kokkos::ALL());
    auto inv_sqrt_lambda_batch = Kokkos::subview(inv_sqrt_lambda_batch_alloc, Kokkos::make_pair(0, (int)n_batch_current), Kokkos::ALL());

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
          for (int j = 0; j < m; j++) {
            PetscScalar z_val      = Z_global_view(obs_idx, j);
            Z_batch(i_local, k, j) = z_val; /* Store Z for potential later use */
            S_batch(i_local, k, j) = (z_val - y_mean_val) * scale_factor;
          }
        }
      });
    Kokkos::fence();

    /* DEBUG: Check S for NaNs */
    if (PetscDefined(USE_DEBUG)) {
      PetscInt nan_count = 0;
      Kokkos::parallel_reduce(
        "CheckS", Kokkos::RangePolicy<exec_space>(0, n_batch_current),
        KOKKOS_LAMBDA(const int i, int &l_count) {
          for (int j = 0; j < n_obs_vertex_copy; j++) {
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
        auto S_i = Kokkos::subview(S_batch, i, Kokkos::ALL(), Kokkos::ALL());
        auto T_i = Kokkos::subview(T_batch, i, Kokkos::ALL(), Kokkos::ALL());

        /* Compute upper triangle of T_i = (1/rho)I + S_i^T * S_i */
        /* T_i(j,k) = (1/rho)*delta_jk + sum_p S_i(p,j) * S_i(p,k) for j <= k */
        for (int j = 0; j < m; j++) {
          for (int k = j; k < m; k++) {
            PetscScalar sum = (j == k) ? inflation_inv : 0.0;
            for (int p = 0; p < n_obs_vertex_copy; p++) sum += S_i(p, j) * S_i(p, k);
            T_i(j, k) = sum;
          }
        }

        /* Copy upper triangle to lower triangle (T is symmetric) */
        for (int j = 0; j < m; j++) {
          for (int k = 0; k < j; k++) T_i(j, k) = T_i(k, j);
        }
      });
    Kokkos::fence();

    /* DEBUG: Check T for NaNs */
    if (PetscDefined(USE_DEBUG)) {
      PetscInt nan_count = 0;
      Kokkos::parallel_reduce(
        "CheckT", Kokkos::RangePolicy<exec_space>(0, n_batch_current),
        KOKKOS_LAMBDA(const int i, int &l_count) {
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
        KOKKOS_LAMBDA(const int i, int &l_count) {
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
        auto S_i               = Kokkos::subview(S_batch, i, Kokkos::ALL(), Kokkos::ALL());
        auto V_i               = Kokkos::subview(V_batch, i, Kokkos::ALL(), Kokkos::ALL());
        auto Lambda_i          = Kokkos::subview(Lambda_batch, i, Kokkos::ALL());
        auto delta_i           = Kokkos::subview(delta_batch, i, Kokkos::ALL());
        auto w_i               = Kokkos::subview(w_batch, i, Kokkos::ALL());
        auto inv_sqrt_lambda_i = Kokkos::subview(inv_sqrt_lambda_batch, i, Kokkos::ALL());
        auto temp1             = Kokkos::subview(temp1_batch, i, Kokkos::ALL());
        auto temp2             = Kokkos::subview(temp2_batch, i, Kokkos::ALL());

        /* 1. Compute w_i = V * L^-1 * V^T * S^T * delta */
        /* Step 1a: temp1 = S^T * delta using KokkosBlas::gemv for better vectorization */
        KokkosBlas::SerialGemv<KokkosBlas::Trans::Transpose, KokkosBlas::Algo::Gemv::Unblocked>::invoke(1.0, S_i, delta_i, 0.0, temp1);

        /* Step 1b: temp2 = V^T * temp1 using KokkosBlas::gemv for better vectorization */
        KokkosBlas::SerialGemv<KokkosBlas::Trans::Transpose, KokkosBlas::Algo::Gemv::Unblocked>::invoke(1.0, V_i, temp1, 0.0, temp2);

        /* Step 1c: temp2 = temp2 / Lambda */
        for (int j = 0; j < m; j++) temp2(j) /= (Lambda_i(j) + 1.0e-14);

        /* Step 1d: w = V * temp2 using KokkosBlas::gemv for better vectorization */
        KokkosBlas::SerialGemv<KokkosBlas::Trans::NoTranspose, KokkosBlas::Algo::Gemv::Unblocked>::invoke(1.0, V_i, temp2, 0.0, w_i);

        /* 2. Precompute 1/sqrt(Lambda) for ensemble update */
        for (int p = 0; p < m; p++) inv_sqrt_lambda_i(p) = 1.0 / Kokkos::sqrt(PetscRealPart(Lambda_i(p)) + 1.0e-14);
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
    Kokkos::deep_copy(dst, e_managed);
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

  /* Ensemble has been updated in batched form above */
  PetscCall(MatAssemblyBegin(en->ensemble, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(en->ensemble, MAT_FINAL_ASSEMBLY));

  {
    MatInfo   info;
    PetscReal flops = 0.0;
    PetscReal n_obs_total;

    if (impl->Q) {
      PetscCall(MatGetInfo(impl->Q, MAT_LOCAL, &info));
      n_obs_total = info.nz_used;
    } else {
      n_obs_total = 0.0;
    }

    /* Step 2.1.2: Fused observation extraction and S/Delta computation */
    flops += n_obs_total * (2.0 + 2.0 * m);

    /* Step 2.1.4: Optimized T matrix formation */
    flops += (PetscReal)n_vertices * m * (m + 1) * impl->n_obs_vertex;

    /* Step 3.1.2: Precompute w and inv_sqrt_lambda */
    flops += (PetscReal)n_vertices * (2.0 * m * impl->n_obs_vertex + 4.0 * m * m + 3.0 * m);

    /* Step 3.1.3: Fused G computation and ensemble update */
    /* T_sqrt: 1.5*m^3 + 1.5*m^2 */
    flops += (PetscReal)n_vertices * (1.5 * m * m * m + 1.5 * m * m);
    /* E update: ndof * m * (4*m + 1) */
    /* Note: G_jk computation (2 flops) is inside the inner loop, so it's 2*m*ndof*m */
    /* Matrix product X*G (2 flops) is also 2*m*ndof*m */
    flops += (PetscReal)n_vertices * ndof * m * (4.0 * m + 1.0);

    PetscCall(PetscLogGpuFlops(flops));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

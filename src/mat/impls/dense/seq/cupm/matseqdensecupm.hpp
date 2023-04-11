#ifndef PETSCMATSEQDENSECUPM_HPP
#define PETSCMATSEQDENSECUPM_HPP

#include <petsc/private/matdensecupmimpl.h> /*I <petscmat.h> I*/
#include <../src/mat/impls/dense/seq/dense.h>

#if defined(__cplusplus)
  #include <petsc/private/deviceimpl.h> // PetscDeviceContextGetOptionalNullContext_Internal()
  #include <petsc/private/randomimpl.h> // _p_PetscRandom
  #include <petsc/private/vecimpl.h>    // _p_Vec
  #include <petsc/private/cupmobject.hpp>
  #include <petsc/private/cupmsolverinterface.hpp>

  #include <petsc/private/cpp/type_traits.hpp> // PetscObjectCast()
  #include <petsc/private/cpp/utility.hpp>     // util::exchange()

  #include <../src/vec/vec/impls/seq/cupm/vecseqcupm.hpp> // for VecSeq_CUPM

namespace Petsc
{

namespace mat
{

namespace cupm
{

namespace impl
{

template <device::cupm::DeviceType T>
class MatDense_Seq_CUPM : MatDense_CUPM<T, MatDense_Seq_CUPM<T>> {
public:
  MATDENSECUPM_HEADER(T, MatDense_Seq_CUPM<T>);

private:
  struct Mat_SeqDenseCUPM {
    PetscScalar *d_v;           // pointer to the matrix on the GPU
    PetscScalar *unplacedarray; // if one called MatCUPMDensePlaceArray(), this is where it stashed the original
    bool         d_user_alloc;
    bool         d_unplaced_user_alloc;
    // factorization support
    cupmBlasInt_t *d_fact_ipiv;  // device pivots
    cupmScalar_t  *d_fact_tau;   // device QR tau vector
    cupmBlasInt_t *d_fact_info;  // device info
    cupmScalar_t  *d_fact_work;  // device workspace
    cupmBlasInt_t  d_fact_lwork; // size of device workspace
    // workspace
    Vec workvec;
  };

  static PetscErrorCode SetPreallocation_(Mat, PetscDeviceContext, PetscScalar *) noexcept;

  static PetscErrorCode HostToDevice_(Mat, PetscDeviceContext) noexcept;
  static PetscErrorCode DeviceToHost_(Mat, PetscDeviceContext) noexcept;

  static PetscErrorCode CheckCUPMSolverInfo_(const cupmBlasInt_t *, cupmStream_t) noexcept;

  template <typename Derived>
  struct SolveCommon;
  struct SolveQR;
  struct SolveCholesky;
  struct SolveLU;

  template <typename Solver, bool transpose>
  static PetscErrorCode MatSolve_Factored_Dispatch_(Mat, Vec, Vec) noexcept;
  template <typename Solver, bool transpose>
  static PetscErrorCode MatMatSolve_Factored_Dispatch_(Mat, Mat, Mat) noexcept;
  template <bool transpose>
  static PetscErrorCode MatMultAdd_Dispatch_(Mat, Vec, Vec, Vec) noexcept;

  template <bool to_host>
  static PetscErrorCode Convert_Dispatch_(Mat, MatType, MatReuse, Mat *) noexcept;

  PETSC_NODISCARD static constexpr MatType       MATIMPLCUPM_() noexcept;
  PETSC_NODISCARD static constexpr Mat_SeqDense *MatIMPLCast_(Mat) noexcept;

public:
  PETSC_NODISCARD static constexpr Mat_SeqDenseCUPM *MatCUPMCast(Mat) noexcept;

  // define these by hand since they don't fit the above mold
  PETSC_NODISCARD static constexpr const char *MatConvert_seqdensecupm_seqdense_C() noexcept;
  PETSC_NODISCARD static constexpr const char *MatProductSetFromOptions_seqaij_seqdensecupm_C() noexcept;

  static PetscErrorCode Create(Mat) noexcept;
  static PetscErrorCode Destroy(Mat) noexcept;
  static PetscErrorCode SetUp(Mat) noexcept;
  static PetscErrorCode Reset(Mat) noexcept;

  static PetscErrorCode BindToCPU(Mat, PetscBool) noexcept;
  static PetscErrorCode Convert_SeqDense_SeqDenseCUPM(Mat, MatType, MatReuse, Mat *) noexcept;
  static PetscErrorCode Convert_SeqDenseCUPM_SeqDense(Mat, MatType, MatReuse, Mat *) noexcept;

  template <PetscMemType, PetscMemoryAccessMode>
  static PetscErrorCode GetArray(Mat, PetscScalar **, PetscDeviceContext) noexcept;
  template <PetscMemType, PetscMemoryAccessMode>
  static PetscErrorCode RestoreArray(Mat, PetscScalar **, PetscDeviceContext) noexcept;
  template <PetscMemoryAccessMode>
  static PetscErrorCode GetArrayAndMemType(Mat, PetscScalar **, PetscMemType *, PetscDeviceContext) noexcept;
  template <PetscMemoryAccessMode>
  static PetscErrorCode RestoreArrayAndMemType(Mat, PetscScalar **, PetscDeviceContext) noexcept;

private:
  template <PetscMemType mtype, PetscMemoryAccessMode mode>
  static PetscErrorCode GetArrayC_(Mat m, PetscScalar **p) noexcept
  {
    PetscDeviceContext dctx;

    PetscFunctionBegin;
    PetscCall(GetHandles_(&dctx));
    PetscCall(GetArray<mtype, mode>(m, p, dctx));
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  template <PetscMemType mtype, PetscMemoryAccessMode mode>
  static PetscErrorCode RestoreArrayC_(Mat m, PetscScalar **p) noexcept
  {
    PetscDeviceContext dctx;

    PetscFunctionBegin;
    PetscCall(GetHandles_(&dctx));
    PetscCall(RestoreArray<mtype, mode>(m, p, dctx));
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  template <PetscMemoryAccessMode mode>
  static PetscErrorCode GetArrayAndMemTypeC_(Mat m, PetscScalar **p, PetscMemType *tp) noexcept
  {
    PetscDeviceContext dctx;

    PetscFunctionBegin;
    PetscCall(GetHandles_(&dctx));
    PetscCall(GetArrayAndMemType<mode>(m, p, tp, dctx));
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  template <PetscMemoryAccessMode mode>
  static PetscErrorCode RestoreArrayAndMemTypeC_(Mat m, PetscScalar **p) noexcept
  {
    PetscDeviceContext dctx;

    PetscFunctionBegin;
    PetscCall(GetHandles_(&dctx));
    PetscCall(RestoreArrayAndMemType<mode>(m, p, dctx));
    PetscFunctionReturn(PETSC_SUCCESS);
  }

public:
  static PetscErrorCode PlaceArray(Mat, const PetscScalar *) noexcept;
  static PetscErrorCode ReplaceArray(Mat, const PetscScalar *) noexcept;
  static PetscErrorCode ResetArray(Mat) noexcept;

  template <bool transpose_A, bool transpose_B>
  static PetscErrorCode MatMatMult_Numeric_Dispatch(Mat, Mat, Mat) noexcept;
  static PetscErrorCode Copy(Mat, Mat, MatStructure) noexcept;
  static PetscErrorCode ZeroEntries(Mat) noexcept;
  static PetscErrorCode Scale(Mat, PetscScalar) noexcept;
  static PetscErrorCode Shift(Mat, PetscScalar) noexcept;
  static PetscErrorCode AXPY(Mat, PetscScalar, Mat, MatStructure) noexcept;
  static PetscErrorCode Duplicate(Mat, MatDuplicateOption, Mat *) noexcept;
  static PetscErrorCode SetRandom(Mat, PetscRandom) noexcept;

  static PetscErrorCode GetColumnVector(Mat, Vec, PetscInt) noexcept;
  template <PetscMemoryAccessMode>
  static PetscErrorCode GetColumnVec(Mat, PetscInt, Vec *) noexcept;
  template <PetscMemoryAccessMode>
  static PetscErrorCode RestoreColumnVec(Mat, PetscInt, Vec *) noexcept;

  static PetscErrorCode GetFactor(Mat, MatFactorType, Mat *) noexcept;
  static PetscErrorCode InvertFactors(Mat) noexcept;

  static PetscErrorCode GetSubMatrix(Mat, PetscInt, PetscInt, PetscInt, PetscInt, Mat *) noexcept;
  static PetscErrorCode RestoreSubMatrix(Mat, Mat *) noexcept;
};

} // namespace impl

namespace
{

// Declare this here so that the functions below can make use of it
template <device::cupm::DeviceType T>
inline PetscErrorCode MatCreateSeqDenseCUPM(MPI_Comm comm, PetscInt m, PetscInt n, PetscScalar *data, Mat *A, PetscDeviceContext dctx = nullptr, bool preallocate = true) noexcept
{
  PetscFunctionBegin;
  PetscCall(impl::MatDense_Seq_CUPM<T>::CreateIMPLDenseCUPM(comm, m, n, m, n, data, A, dctx, preallocate));
  PetscFunctionReturn(PETSC_SUCCESS);
}

} // anonymous namespace

namespace impl
{

// ==========================================================================================
// MatDense_Seq_CUPM - Private API - Utility
// ==========================================================================================

template <device::cupm::DeviceType T>
inline PetscErrorCode MatDense_Seq_CUPM<T>::SetPreallocation_(Mat m, PetscDeviceContext dctx, PetscScalar *user_device_array) noexcept
{
  const auto   mcu   = MatCUPMCast(m);
  const auto   nrows = m->rmap->n;
  const auto   ncols = m->cmap->n;
  auto        &lda   = MatIMPLCast(m)->lda;
  cupmStream_t stream;

  PetscFunctionBegin;
  PetscCheckTypeName(m, MATSEQDENSECUPM());
  PetscValidDeviceContext(dctx, 2);
  PetscCall(checkCupmBlasIntCast(nrows));
  PetscCall(checkCupmBlasIntCast(ncols));
  PetscCall(GetHandlesFrom_(dctx, &stream));
  if (lda <= 0) lda = nrows;
  if (!mcu->d_user_alloc) PetscCallCUPM(cupmFreeAsync(mcu->d_v, stream));
  if (user_device_array) {
    mcu->d_user_alloc = PETSC_TRUE;
    mcu->d_v          = user_device_array;
  } else {
    PetscInt size;

    mcu->d_user_alloc = PETSC_FALSE;
    PetscCall(PetscIntMultError(lda, ncols, &size));
    PetscCall(PetscCUPMMallocAsync(&mcu->d_v, size, stream));
    PetscCall(PetscCUPMMemsetAsync(mcu->d_v, 0, size, stream));
  }
  m->offloadmask = PETSC_OFFLOAD_GPU;
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <device::cupm::DeviceType T>
inline PetscErrorCode MatDense_Seq_CUPM<T>::HostToDevice_(Mat m, PetscDeviceContext dctx) noexcept
{
  const auto nrows = m->rmap->n;
  const auto ncols = m->cmap->n;
  const auto copy  = m->offloadmask == PETSC_OFFLOAD_CPU || m->offloadmask == PETSC_OFFLOAD_UNALLOCATED;

  PetscFunctionBegin;
  PetscCheckTypeName(m, MATSEQDENSECUPM());
  if (m->boundtocpu) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(PetscInfo(m, "%s matrix %" PetscInt_FMT " x %" PetscInt_FMT "\n", copy ? "Copy" : "Reusing", nrows, ncols));
  if (copy) {
    const auto   mcu = MatCUPMCast(m);
    cupmStream_t stream;

    // Allocate GPU memory if not present
    if (!mcu->d_v) PetscCall(SetPreallocation(m, dctx));
    PetscCall(GetHandlesFrom_(dctx, &stream));
    PetscCall(PetscLogEventBegin(MAT_DenseCopyToGPU, m, 0, 0, 0));
    {
      const auto mimpl = MatIMPLCast(m);
      const auto lda   = mimpl->lda;
      const auto src   = mimpl->v;
      const auto dest  = mcu->d_v;

      if (lda > nrows) {
        PetscCall(PetscCUPMMemcpy2DAsync(dest, lda, src, lda, nrows, ncols, cupmMemcpyHostToDevice, stream));
      } else {
        PetscCall(PetscCUPMMemcpyAsync(dest, src, lda * ncols, cupmMemcpyHostToDevice, stream));
      }
    }
    PetscCall(PetscLogEventEnd(MAT_DenseCopyToGPU, m, 0, 0, 0));
    // order important, ensure that offloadmask is PETSC_OFFLOAD_BOTH
    m->offloadmask = PETSC_OFFLOAD_BOTH;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <device::cupm::DeviceType T>
inline PetscErrorCode MatDense_Seq_CUPM<T>::DeviceToHost_(Mat m, PetscDeviceContext dctx) noexcept
{
  const auto nrows = m->rmap->n;
  const auto ncols = m->cmap->n;
  const auto copy  = m->offloadmask == PETSC_OFFLOAD_GPU;

  PetscFunctionBegin;
  PetscCheckTypeName(m, MATSEQDENSECUPM());
  PetscCall(PetscInfo(m, "%s matrix %" PetscInt_FMT " x %" PetscInt_FMT "\n", copy ? "Copy" : "Reusing", nrows, ncols));
  if (copy) {
    const auto   mimpl = MatIMPLCast(m);
    cupmStream_t stream;

    // MatCreateSeqDenseCUPM may not allocate CPU memory. Allocate if needed
    if (!mimpl->v) PetscCall(MatSeqDenseSetPreallocation(m, nullptr));
    PetscCall(GetHandlesFrom_(dctx, &stream));
    PetscCall(PetscLogEventBegin(MAT_DenseCopyFromGPU, m, 0, 0, 0));
    {
      const auto lda  = mimpl->lda;
      const auto dest = mimpl->v;
      const auto src  = MatCUPMCast(m)->d_v;

      if (lda > nrows) {
        PetscCall(PetscCUPMMemcpy2DAsync(dest, lda, src, lda, nrows, ncols, cupmMemcpyDeviceToHost, stream));
      } else {
        PetscCall(PetscCUPMMemcpyAsync(dest, src, lda * ncols, cupmMemcpyDeviceToHost, stream));
      }
    }
    PetscCall(PetscLogEventEnd(MAT_DenseCopyFromGPU, m, 0, 0, 0));
    // order is important, MatSeqDenseSetPreallocation() might set offloadmask
    m->offloadmask = PETSC_OFFLOAD_BOTH;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <device::cupm::DeviceType T>
inline PetscErrorCode MatDense_Seq_CUPM<T>::CheckCUPMSolverInfo_(const cupmBlasInt_t *fact_info, cupmStream_t stream) noexcept
{
  PetscFunctionBegin;
  if (PetscDefined(USE_DEBUG)) {
    cupmBlasInt_t info = 0;

    PetscCall(PetscCUPMMemcpyAsync(&info, fact_info, 1, cupmMemcpyDeviceToHost, stream));
    if (stream) PetscCallCUPM(cupmStreamSynchronize(stream));
    static_assert(std::is_same<decltype(info), int>::value, "");
    PetscCheck(info <= 0, PETSC_COMM_SELF, PETSC_ERR_MAT_CH_ZRPVT, "Bad factorization: zero pivot in row %d", info - 1);
    PetscCheck(info >= 0, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Wrong argument to cupmSolver %d", -info);
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

// ==========================================================================================
// MatDense_Seq_CUPM - Private API - Solver Dispatch
// ==========================================================================================

// specific solvers called through the dispatch_() family of functions
template <device::cupm::DeviceType T>
template <typename Derived>
struct MatDense_Seq_CUPM<T>::SolveCommon {
  using derived_type = Derived;

  template <typename F>
  static PetscErrorCode ResizeFactLwork(Mat_SeqDenseCUPM *mcu, cupmStream_t stream, F &&cupmSolverComputeFactLwork) noexcept
  {
    cupmBlasInt_t lwork;

    PetscFunctionBegin;
    PetscCallCUPMSOLVER(cupmSolverComputeFactLwork(&lwork));
    if (lwork > mcu->d_fact_lwork) {
      mcu->d_fact_lwork = lwork;
      PetscCallCUPM(cupmFreeAsync(mcu->d_fact_work, stream));
      PetscCall(PetscCUPMMallocAsync(&mcu->d_fact_work, lwork, stream));
    }
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  static PetscErrorCode FactorPrepare(Mat A, cupmStream_t stream) noexcept
  {
    const auto mcu = MatCUPMCast(A);

    PetscFunctionBegin;
    PetscCall(PetscInfo(A, "%s factor %" PetscInt_FMT " x %" PetscInt_FMT " on backend\n", derived_type::NAME(), A->rmap->n, A->cmap->n));
    A->factortype             = derived_type::MATFACTORTYPE();
    A->ops->solve             = MatSolve_Factored_Dispatch_<derived_type, false>;
    A->ops->solvetranspose    = MatSolve_Factored_Dispatch_<derived_type, true>;
    A->ops->matsolve          = MatMatSolve_Factored_Dispatch_<derived_type, false>;
    A->ops->matsolvetranspose = MatMatSolve_Factored_Dispatch_<derived_type, true>;

    PetscCall(PetscStrFreeAllocpy(MATSOLVERCUPM(), &A->solvertype));
    if (!mcu->d_fact_info) PetscCall(PetscCUPMMallocAsync(&mcu->d_fact_info, 1, stream));
    PetscFunctionReturn(PETSC_SUCCESS);
  }
};

template <device::cupm::DeviceType T>
struct MatDense_Seq_CUPM<T>::SolveLU : SolveCommon<SolveLU> {
  using base_type = SolveCommon<SolveLU>;

  static constexpr const char   *NAME() noexcept { return "LU"; }
  static constexpr MatFactorType MATFACTORTYPE() noexcept { return MAT_FACTOR_LU; }

  static PetscErrorCode Factor(Mat A, IS, IS, const MatFactorInfo *) noexcept
  {
    const auto         m = static_cast<cupmBlasInt_t>(A->rmap->n);
    const auto         n = static_cast<cupmBlasInt_t>(A->cmap->n);
    cupmStream_t       stream;
    cupmSolverHandle_t handle;
    PetscDeviceContext dctx;

    PetscFunctionBegin;
    if (!m || !n) PetscFunctionReturn(PETSC_SUCCESS);
    PetscCall(GetHandles_(&dctx, &handle, &stream));
    PetscCall(base_type::FactorPrepare(A, stream));
    {
      const auto mcu = MatCUPMCast(A);
      const auto da  = DeviceArrayReadWrite(dctx, A);
      const auto lda = static_cast<cupmBlasInt_t>(MatIMPLCast(A)->lda);

      // clang-format off
      PetscCall(
        base_type::ResizeFactLwork(
          mcu, stream,
          [&](cupmBlasInt_t *fact_lwork)
          {
            return cupmSolverXgetrf_bufferSize(handle, m, n, da.cupmdata(), lda, fact_lwork);
          }
        )
      );
      // clang-format on
      if (!mcu->d_fact_ipiv) PetscCall(PetscCUPMMallocAsync(&mcu->d_fact_ipiv, n, stream));

      PetscCall(PetscLogGpuTimeBegin());
      PetscCallCUPMSOLVER(cupmSolverXgetrf(handle, m, n, da.cupmdata(), lda, mcu->d_fact_work, mcu->d_fact_lwork, mcu->d_fact_ipiv, mcu->d_fact_info));
      PetscCall(PetscLogGpuTimeEnd());
      PetscCall(CheckCUPMSolverInfo_(mcu->d_fact_info, stream));
    }
    PetscCall(PetscLogGpuFlops(2.0 * n * n * m / 3.0));
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  template <bool transpose>
  static PetscErrorCode Solve(Mat A, cupmScalar_t *x, cupmBlasInt_t ldx, cupmBlasInt_t m, cupmBlasInt_t nrhs, cupmBlasInt_t k, PetscDeviceContext dctx, cupmStream_t stream) noexcept
  {
    const auto         mcu       = MatCUPMCast(A);
    const auto         fact_info = mcu->d_fact_info;
    const auto         fact_ipiv = mcu->d_fact_ipiv;
    cupmSolverHandle_t handle;

    PetscFunctionBegin;
    PetscCall(GetHandlesFrom_(dctx, &handle));
    PetscCall(PetscInfo(A, "%s solve %d x %d on backend\n", NAME(), m, k));
    PetscCall(PetscLogGpuTimeBegin());
    {
      constexpr auto op  = transpose ? CUPMSOLVER_OP_T : CUPMSOLVER_OP_N;
      const auto     da  = DeviceArrayRead(dctx, A);
      const auto     lda = static_cast<cupmBlasInt_t>(MatIMPLCast(A)->lda);

      // clang-format off
      PetscCall(
        base_type::ResizeFactLwork(
          mcu, stream,
          [&](cupmBlasInt_t *lwork)
          {
            return cupmSolverXgetrs_bufferSize(
              handle, op, m, nrhs, da.cupmdata(), lda, fact_ipiv, x, ldx, lwork
            );
          }
        )
      );
      // clang-format on
      PetscCallCUPMSOLVER(cupmSolverXgetrs(handle, op, m, nrhs, da.cupmdata(), lda, fact_ipiv, x, ldx, mcu->d_fact_work, mcu->d_fact_lwork, fact_info));
      PetscCall(CheckCUPMSolverInfo_(fact_info, stream));
    }
    PetscCall(PetscLogGpuTimeEnd());
    PetscCall(PetscLogGpuFlops(nrhs * (2.0 * m * m - m)));
    PetscFunctionReturn(PETSC_SUCCESS);
  }
};

template <device::cupm::DeviceType T>
struct MatDense_Seq_CUPM<T>::SolveCholesky : SolveCommon<SolveCholesky> {
  using base_type = SolveCommon<SolveCholesky>;

  static constexpr const char   *NAME() noexcept { return "Cholesky"; }
  static constexpr MatFactorType MATFACTORTYPE() noexcept { return MAT_FACTOR_CHOLESKY; }

  static PetscErrorCode Factor(Mat A, IS, const MatFactorInfo *) noexcept
  {
    const auto         n = static_cast<cupmBlasInt_t>(A->rmap->n);
    PetscDeviceContext dctx;
    cupmSolverHandle_t handle;
    cupmStream_t       stream;

    PetscFunctionBegin;
    if (!n || !A->cmap->n) PetscFunctionReturn(PETSC_SUCCESS);
    PetscCheck(A->spd == PETSC_BOOL3_TRUE, PETSC_COMM_SELF, PETSC_ERR_SUP, "%ssytrs unavailable. Use MAT_FACTOR_LU", cupmSolverName());
    PetscCall(GetHandles_(&dctx, &handle, &stream));
    PetscCall(base_type::FactorPrepare(A, stream));
    {
      const auto mcu = MatCUPMCast(A);
      const auto da  = DeviceArrayReadWrite(dctx, A);
      const auto lda = static_cast<cupmBlasInt_t>(MatIMPLCast(A)->lda);

      // clang-format off
      PetscCall(
        base_type::ResizeFactLwork(
          mcu, stream,
          [&](cupmBlasInt_t *fact_lwork)
          {
            return cupmSolverXpotrf_bufferSize(
              handle, CUPMSOLVER_FILL_MODE_LOWER, n, da.cupmdata(), lda, fact_lwork
            );
          }
        )
      );
      // clang-format on
      PetscCall(PetscLogGpuTimeBegin());
      PetscCallCUPMSOLVER(cupmSolverXpotrf(handle, CUPMSOLVER_FILL_MODE_LOWER, n, da.cupmdata(), lda, mcu->d_fact_work, mcu->d_fact_lwork, mcu->d_fact_info));
      PetscCall(PetscLogGpuTimeEnd());
      PetscCall(CheckCUPMSolverInfo_(mcu->d_fact_info, stream));
    }
    PetscCall(PetscLogGpuFlops(1.0 * n * n * n / 3.0));

  #if 0
    // At the time of writing this interface (cuda 10.0), cusolverDn does not implement *sytrs
    // and *hetr* routines. The code below should work, and it can be activated when *sytrs
    // routines will be available
    if (!mcu->d_fact_ipiv) PetscCall(PetscCUPMMallocAsync(&mcu->d_fact_ipiv, n, stream));
    if (!mcu->d_fact_lwork) {
      PetscCallCUPMSOLVER(cupmSolverDnXsytrf_bufferSize(handle, n, da.cupmdata(), lda, &mcu->d_fact_lwork));
      PetscCall(PetscCUPMMallocAsync(&mcu->d_fact_work, mcu->d_fact_lwork, stream));
    }
    if (mcu->d_fact_info) PetscCall(PetscCUPMMallocAsync(&mcu->d_fact_info, 1, stream));
    PetscCall(PetscLogGpuTimeBegin());
    PetscCallCUPMSOLVER(cupmSolverXsytrf(handle, CUPMSOLVER_FILL_MODE_LOWER, n, da, lda, mcu->d_fact_ipiv, mcu->d_fact_work, mcu->d_fact_lwork, mcu->d_fact_info));
    PetscCall(PetscLogGpuTimeEnd());
  #endif
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  template <bool transpose>
  static PetscErrorCode Solve(Mat A, cupmScalar_t *x, cupmBlasInt_t ldx, cupmBlasInt_t m, cupmBlasInt_t nrhs, cupmBlasInt_t k, PetscDeviceContext dctx, cupmStream_t stream) noexcept
  {
    const auto         mcu       = MatCUPMCast(A);
    const auto         fact_info = mcu->d_fact_info;
    cupmSolverHandle_t handle;

    PetscFunctionBegin;
    PetscAssert(!mcu->d_fact_ipiv, PETSC_COMM_SELF, PETSC_ERR_LIB, "%ssytrs not implemented", cupmSolverName());
    PetscCall(GetHandlesFrom_(dctx, &handle));
    PetscCall(PetscInfo(A, "%s solve %d x %d on backend\n", NAME(), m, k));
    PetscCall(PetscLogGpuTimeBegin());
    {
      const auto da  = DeviceArrayRead(dctx, A);
      const auto lda = static_cast<cupmBlasInt_t>(MatIMPLCast(A)->lda);

      // clang-format off
      PetscCall(
        base_type::ResizeFactLwork(
          mcu, stream,
          [&](cupmBlasInt_t *lwork)
          {
            return cupmSolverXpotrs_bufferSize(
              handle, CUPMSOLVER_FILL_MODE_LOWER, m, nrhs, da.cupmdata(), lda, x, ldx, lwork
            );
          }
        )
      );
      // clang-format on
      PetscCallCUPMSOLVER(cupmSolverXpotrs(handle, CUPMSOLVER_FILL_MODE_LOWER, m, nrhs, da.cupmdata(), lda, x, ldx, mcu->d_fact_work, mcu->d_fact_lwork, fact_info));
      PetscCall(CheckCUPMSolverInfo_(fact_info, stream));
    }
    PetscCall(PetscLogGpuTimeEnd());
    PetscCall(PetscLogGpuFlops(nrhs * (2.0 * m * m - m)));
    PetscFunctionReturn(PETSC_SUCCESS);
  }
};

template <device::cupm::DeviceType T>
struct MatDense_Seq_CUPM<T>::SolveQR : SolveCommon<SolveQR> {
  using base_type = SolveCommon<SolveQR>;

  static constexpr const char   *NAME() noexcept { return "QR"; }
  static constexpr MatFactorType MATFACTORTYPE() noexcept { return MAT_FACTOR_QR; }

  static PetscErrorCode Factor(Mat A, IS, const MatFactorInfo *) noexcept
  {
    const auto         m     = static_cast<cupmBlasInt_t>(A->rmap->n);
    const auto         n     = static_cast<cupmBlasInt_t>(A->cmap->n);
    const auto         min   = std::min(m, n);
    const auto         mimpl = MatIMPLCast(A);
    cupmStream_t       stream;
    cupmSolverHandle_t handle;
    PetscDeviceContext dctx;

    PetscFunctionBegin;
    if (!m || !n) PetscFunctionReturn(PETSC_SUCCESS);
    PetscCall(GetHandles_(&dctx, &handle, &stream));
    PetscCall(base_type::FactorPrepare(A, stream));
    mimpl->rank = min;
    {
      const auto mcu = MatCUPMCast(A);
      const auto da  = DeviceArrayReadWrite(dctx, A);
      const auto lda = static_cast<cupmBlasInt_t>(mimpl->lda);

      if (!mcu->workvec) PetscCall(vec::cupm::VecCreateSeqCUPMAsync<T>(PetscObjectComm(PetscObjectCast(A)), m, &mcu->workvec));
      if (!mcu->d_fact_tau) PetscCall(PetscCUPMMallocAsync(&mcu->d_fact_tau, min, stream));
      // clang-format off
      PetscCall(
        base_type::ResizeFactLwork(
          mcu, stream,
          [&](cupmBlasInt_t *fact_lwork)
          {
            return cupmSolverXgeqrf_bufferSize(handle, m, n, da.cupmdata(), lda, fact_lwork);
          }
        )
      );
      // clang-format on
      PetscCall(PetscLogGpuTimeBegin());
      PetscCallCUPMSOLVER(cupmSolverXgeqrf(handle, m, n, da.cupmdata(), lda, mcu->d_fact_tau, mcu->d_fact_work, mcu->d_fact_lwork, mcu->d_fact_info));
      PetscCall(PetscLogGpuTimeEnd());
      PetscCall(CheckCUPMSolverInfo_(mcu->d_fact_info, stream));
    }
    PetscCall(PetscLogGpuFlops(2.0 * min * min * (std::max(m, n) - min / 3.0)));
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  template <bool transpose>
  static PetscErrorCode Solve(Mat A, cupmScalar_t *x, cupmBlasInt_t ldx, cupmBlasInt_t m, cupmBlasInt_t nrhs, cupmBlasInt_t k, PetscDeviceContext dctx, cupmStream_t stream) noexcept
  {
    const auto         mimpl      = MatIMPLCast(A);
    const auto         rank       = static_cast<cupmBlasInt_t>(mimpl->rank);
    const auto         mcu        = MatCUPMCast(A);
    const auto         fact_info  = mcu->d_fact_info;
    const auto         fact_tau   = mcu->d_fact_tau;
    const auto         fact_work  = mcu->d_fact_work;
    const auto         fact_lwork = mcu->d_fact_lwork;
    cupmSolverHandle_t solver_handle;
    cupmBlasHandle_t   blas_handle;

    PetscFunctionBegin;
    PetscCall(GetHandlesFrom_(dctx, &blas_handle, &solver_handle));
    PetscCall(PetscInfo(A, "%s solve %d x %d on backend\n", NAME(), m, k));
    PetscCall(PetscLogGpuTimeBegin());
    {
      const auto da  = DeviceArrayRead(dctx, A);
      const auto one = cupmScalarCast(1.0);
      const auto lda = static_cast<cupmBlasInt_t>(mimpl->lda);

      if (transpose) {
        PetscCallCUPMBLAS(cupmBlasXtrsm(blas_handle, CUPMBLAS_SIDE_LEFT, CUPMBLAS_FILL_MODE_UPPER, CUPMBLAS_OP_T, CUPMBLAS_DIAG_NON_UNIT, rank, nrhs, &one, da.cupmdata(), lda, x, ldx));
        PetscCallCUPMSOLVER(cupmSolverXormqr(solver_handle, CUPMSOLVER_SIDE_LEFT, CUPMSOLVER_OP_N, m, nrhs, rank, da.cupmdata(), lda, fact_tau, x, ldx, fact_work, fact_lwork, fact_info));
        PetscCall(CheckCUPMSolverInfo_(fact_info, stream));
      } else {
        constexpr auto op = PetscDefined(USE_COMPLEX) ? CUPMSOLVER_OP_C : CUPMSOLVER_OP_T;

        PetscCallCUPMSOLVER(cupmSolverXormqr(solver_handle, CUPMSOLVER_SIDE_LEFT, op, m, nrhs, rank, da.cupmdata(), lda, fact_tau, x, ldx, fact_work, fact_lwork, fact_info));
        PetscCall(CheckCUPMSolverInfo_(fact_info, stream));
        PetscCallCUPMBLAS(cupmBlasXtrsm(blas_handle, CUPMBLAS_SIDE_LEFT, CUPMBLAS_FILL_MODE_UPPER, CUPMBLAS_OP_N, CUPMBLAS_DIAG_NON_UNIT, rank, nrhs, &one, da.cupmdata(), lda, x, ldx));
      }
    }
    PetscCall(PetscLogGpuTimeEnd());
    PetscCall(PetscLogFlops(nrhs * (4.0 * m * rank - (rank * rank))));
    PetscFunctionReturn(PETSC_SUCCESS);
  }
};

template <device::cupm::DeviceType T>
template <typename Solver, bool transpose>
inline PetscErrorCode MatDense_Seq_CUPM<T>::MatSolve_Factored_Dispatch_(Mat A, Vec x, Vec y) noexcept
{
  using namespace vec::cupm;
  const auto         pobj_A  = PetscObjectCast(A);
  const auto         m       = static_cast<cupmBlasInt_t>(A->rmap->n);
  const auto         k       = static_cast<cupmBlasInt_t>(A->cmap->n);
  auto              &workvec = MatCUPMCast(A)->workvec;
  PetscScalar       *y_array = nullptr;
  PetscDeviceContext dctx;
  PetscBool          xiscupm, yiscupm, aiscupm;
  bool               use_y_array_directly;
  cupmStream_t       stream;

  PetscFunctionBegin;
  PetscCheck(A->factortype != MAT_FACTOR_NONE, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "Matrix must be factored to solve");
  PetscCall(PetscObjectTypeCompare(PetscObjectCast(x), VecSeq_CUPM::VECSEQCUPM(), &xiscupm));
  PetscCall(PetscObjectTypeCompare(PetscObjectCast(y), VecSeq_CUPM::VECSEQCUPM(), &yiscupm));
  PetscCall(PetscObjectTypeCompare(pobj_A, MATSEQDENSECUPM(), &aiscupm));
  PetscAssert(aiscupm, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Matrix A is somehow not CUPM?????????????????????????????");
  PetscCall(GetHandles_(&dctx, &stream));
  use_y_array_directly = yiscupm && (k >= m);
  {
    const PetscScalar *x_array;
    const auto         xisdevice = xiscupm && PetscOffloadDevice(x->offloadmask);
    const auto         copy_mode = xisdevice ? cupmMemcpyDeviceToDevice : cupmMemcpyHostToDevice;

    if (!use_y_array_directly && !workvec) PetscCall(VecCreateSeqCUPMAsync<T>(PetscObjectComm(pobj_A), m, &workvec));
    // The logic here is to try to minimize the amount of memory copying:
    //
    // If we call VecCUPMGetArrayRead(X, &x) every time xiscupm and the data is not offloaded
    // to the GPU yet, then the data is copied to the GPU. But we are only trying to get the
    // data in order to copy it into the y array. So the array x will be wherever the data
    // already is so that only one memcpy is performed
    if (xisdevice) {
      PetscCall(VecCUPMGetArrayReadAsync<T>(x, &x_array, dctx));
    } else {
      PetscCall(VecGetArrayRead(x, &x_array));
    }
    PetscCall(VecCUPMGetArrayWriteAsync<T>(use_y_array_directly ? y : workvec, &y_array, dctx));
    PetscCall(PetscCUPMMemcpyAsync(y_array, x_array, m, copy_mode, stream));
    if (xisdevice) {
      PetscCall(VecCUPMRestoreArrayReadAsync<T>(x, &x_array, dctx));
    } else {
      PetscCall(VecRestoreArrayRead(x, &x_array));
    }
  }

  if (!aiscupm) PetscCall(MatConvert(A, MATSEQDENSECUPM(), MAT_INPLACE_MATRIX, &A));
  PetscCall(Solver{}.template Solve<transpose>(A, cupmScalarPtrCast(y_array), m, m, 1, k, dctx, stream));
  if (!aiscupm) PetscCall(MatConvert(A, MATSEQDENSE, MAT_INPLACE_MATRIX, &A));

  if (use_y_array_directly) {
    PetscCall(VecCUPMRestoreArrayWriteAsync<T>(y, &y_array, dctx));
  } else {
    const auto   copy_mode = yiscupm ? cupmMemcpyDeviceToDevice : cupmMemcpyDeviceToHost;
    PetscScalar *yv;

    // The logic here is that the data is not yet in either y's GPU array or its CPU array.
    // There is nothing in the interface to say where the user would like it to end up. So we
    // choose the GPU, because it is the faster option
    if (yiscupm) {
      PetscCall(VecCUPMGetArrayWriteAsync<T>(y, &yv, dctx));
    } else {
      PetscCall(VecGetArray(y, &yv));
    }
    PetscCall(PetscCUPMMemcpyAsync(yv, y_array, k, copy_mode, stream));
    if (yiscupm) {
      PetscCall(VecCUPMRestoreArrayWriteAsync<T>(y, &yv, dctx));
    } else {
      PetscCall(VecRestoreArray(y, &yv));
    }
    PetscCall(VecCUPMRestoreArrayWriteAsync<T>(workvec, &y_array));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <device::cupm::DeviceType T>
template <typename Solver, bool transpose>
inline PetscErrorCode MatDense_Seq_CUPM<T>::MatMatSolve_Factored_Dispatch_(Mat A, Mat B, Mat X) noexcept
{
  const auto         m = static_cast<cupmBlasInt_t>(A->rmap->n);
  const auto         k = static_cast<cupmBlasInt_t>(A->cmap->n);
  cupmBlasInt_t      nrhs, ldb, ldx, ldy;
  PetscScalar       *y;
  PetscBool          biscupm, xiscupm, aiscupm;
  PetscDeviceContext dctx;
  cupmStream_t       stream;

  PetscFunctionBegin;
  PetscCheck(A->factortype != MAT_FACTOR_NONE, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "Matrix must be factored to solve");
  PetscCall(PetscObjectTypeCompare(PetscObjectCast(B), MATSEQDENSECUPM(), &biscupm));
  PetscCall(PetscObjectTypeCompare(PetscObjectCast(X), MATSEQDENSECUPM(), &xiscupm));
  PetscCall(PetscObjectTypeCompare(PetscObjectCast(A), MATSEQDENSECUPM(), &aiscupm));
  PetscCall(GetHandles_(&dctx, &stream));
  {
    PetscInt n;

    PetscCall(MatGetSize(B, nullptr, &n));
    PetscCall(PetscCUPMBlasIntCast(n, &nrhs));
    PetscCall(MatDenseGetLDA(B, &n));
    PetscCall(PetscCUPMBlasIntCast(n, &ldb));
    PetscCall(MatDenseGetLDA(X, &n));
    PetscCall(PetscCUPMBlasIntCast(n, &ldx));
  }
  {
    // The logic here is to try to minimize the amount of memory copying:
    //
    // If we call MatDenseCUPMGetArrayRead(B, &b) every time biscupm and the data is not
    // offloaded to the GPU yet, then the data is copied to the GPU. But we are only trying to
    // get the data in order to copy it into the y array. So the array b will be wherever the
    // data already is so that only one memcpy is performed
    const auto         bisdevice = biscupm && PetscOffloadDevice(B->offloadmask);
    const auto         copy_mode = bisdevice ? cupmMemcpyDeviceToDevice : cupmMemcpyHostToDevice;
    const PetscScalar *b;

    if (bisdevice) {
      b = DeviceArrayRead(dctx, B);
    } else if (biscupm) {
      b = HostArrayRead(dctx, B);
    } else {
      PetscCall(MatDenseGetArrayRead(B, &b));
    }

    if (ldx < m || !xiscupm) {
      // X's array cannot serve as the array (too small or not on device), B's array cannot
      // serve as the array (const), so allocate a new array
      ldy = m;
      PetscCall(PetscCUPMMallocAsync(&y, nrhs * m));
    } else {
      // X's array should serve as the array
      ldy = ldx;
      y   = DeviceArrayWrite(dctx, X);
    }
    PetscCall(PetscCUPMMemcpy2DAsync(y, ldy, b, ldb, m, nrhs, copy_mode, stream));
    if (!bisdevice && !biscupm) PetscCall(MatDenseRestoreArrayRead(B, &b));
  }

  // convert to CUPM twice??????????????????????????????????
  // but A should already be CUPM??????????????????????????????????????
  if (!aiscupm) PetscCall(MatConvert(A, MATSEQDENSECUPM(), MAT_INPLACE_MATRIX, &A));
  PetscCall(Solver{}.template Solve<transpose>(A, cupmScalarPtrCast(y), ldy, m, nrhs, k, dctx, stream));
  if (!aiscupm) PetscCall(MatConvert(A, MATSEQDENSECUPM(), MAT_INPLACE_MATRIX, &A));

  if (ldx < m || !xiscupm) {
    const auto   copy_mode = xiscupm ? cupmMemcpyDeviceToDevice : cupmMemcpyDeviceToHost;
    PetscScalar *x;

    // The logic here is that the data is not yet in either X's GPU array or its CPU
    // array. There is nothing in the interface to say where the user would like it to end up.
    // So we choose the GPU, because it is the faster option
    if (xiscupm) {
      x = DeviceArrayWrite(dctx, X);
    } else {
      PetscCall(MatDenseGetArray(X, &x));
    }
    PetscCall(PetscCUPMMemcpy2DAsync(x, ldx, y, ldy, k, nrhs, copy_mode, stream));
    if (!xiscupm) PetscCall(MatDenseRestoreArray(X, &x));
    PetscCallCUPM(cupmFreeAsync(y, stream));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <device::cupm::DeviceType T>
template <bool transpose>
inline PetscErrorCode MatDense_Seq_CUPM<T>::MatMultAdd_Dispatch_(Mat A, Vec xx, Vec yy, Vec zz) noexcept
{
  const auto         m = static_cast<cupmBlasInt_t>(A->rmap->n);
  const auto         n = static_cast<cupmBlasInt_t>(A->cmap->n);
  cupmBlasHandle_t   handle;
  PetscDeviceContext dctx;

  PetscFunctionBegin;
  if (yy && yy != zz) PetscCall(VecSeq_CUPM::Copy(yy, zz)); // mult add
  if (!m || !n) {
    // mult only
    if (!yy) PetscCall(VecSeq_CUPM::Set(zz, 0.0));
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  PetscCall(PetscInfo(A, "Matrix-vector product %" PetscBLASInt_FMT " x %" PetscBLASInt_FMT " on backend\n", m, n));
  PetscCall(GetHandles_(&dctx, &handle));
  {
    constexpr auto op   = transpose ? CUPMBLAS_OP_T : CUPMBLAS_OP_N;
    const auto     one  = cupmScalarCast(1.0);
    const auto     zero = cupmScalarCast(0.0);
    const auto     da   = DeviceArrayRead(dctx, A);
    const auto     dxx  = VecSeq_CUPM::DeviceArrayRead(dctx, xx);
    const auto     dzz  = VecSeq_CUPM::DeviceArrayReadWrite(dctx, zz);

    PetscCall(PetscLogGpuTimeBegin());
    PetscCallCUPMBLAS(cupmBlasXgemv(handle, op, m, n, &one, da.cupmdata(), static_cast<cupmBlasInt_t>(MatIMPLCast(A)->lda), dxx.cupmdata(), 1, (yy ? &one : &zero), dzz.cupmdata(), 1));
    PetscCall(PetscLogGpuTimeEnd());
  }
  PetscCall(PetscLogGpuFlops(2.0 * m * n - (yy ? 0 : m)));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// ==========================================================================================
// MatDense_Seq_CUPM - Private API - Conversion Dispatch
// ==========================================================================================

template <device::cupm::DeviceType T>
template <bool to_host>
inline PetscErrorCode MatDense_Seq_CUPM<T>::Convert_Dispatch_(Mat M, MatType type, MatReuse reuse, Mat *newmat) noexcept
{
  PetscFunctionBegin;
  if (reuse == MAT_REUSE_MATRIX || reuse == MAT_INITIAL_MATRIX) {
    // TODO these cases should be optimized
    PetscCall(MatConvert_Basic(M, type, reuse, newmat));
  } else {
    const auto B    = *newmat;
    const auto pobj = PetscObjectCast(B);

    if (to_host) {
      PetscCall(BindToCPU(B, PETSC_TRUE));
      PetscCall(Reset(B));
    } else {
      PetscCall(PetscDeviceInitialize(PETSC_DEVICE_CUPM()));
    }

    PetscCall(PetscStrFreeAllocpy(to_host ? VECSTANDARD : VecSeq_CUPM::VECCUPM(), &B->defaultvectype));
    PetscCall(PetscObjectChangeTypeName(pobj, to_host ? MATSEQDENSE : MATSEQDENSECUPM()));
    // cvec might be the wrong VecType, destroy and rebuild it if necessary
    // REVIEW ME: this is possibly very inefficient
    PetscCall(VecDestroy(&MatIMPLCast(B)->cvec));

    MatComposeOp_CUPM(to_host, pobj, MatConvert_seqdensecupm_seqdense_C(), nullptr, Convert_SeqDenseCUPM_SeqDense);
    MatComposeOp_CUPM(to_host, pobj, MatDenseCUPMGetArray_C(), nullptr, GetArrayC_<PETSC_MEMTYPE_DEVICE, PETSC_MEMORY_ACCESS_READ_WRITE>);
    MatComposeOp_CUPM(to_host, pobj, MatDenseCUPMGetArrayRead_C(), nullptr, GetArrayC_<PETSC_MEMTYPE_DEVICE, PETSC_MEMORY_ACCESS_READ>);
    MatComposeOp_CUPM(to_host, pobj, MatDenseCUPMGetArrayWrite_C(), nullptr, GetArrayC_<PETSC_MEMTYPE_DEVICE, PETSC_MEMORY_ACCESS_WRITE>);
    MatComposeOp_CUPM(to_host, pobj, MatDenseCUPMRestoreArray_C(), nullptr, RestoreArrayC_<PETSC_MEMTYPE_DEVICE, PETSC_MEMORY_ACCESS_READ_WRITE>);
    MatComposeOp_CUPM(to_host, pobj, MatDenseCUPMRestoreArrayRead_C(), nullptr, RestoreArrayC_<PETSC_MEMTYPE_DEVICE, PETSC_MEMORY_ACCESS_READ>);
    MatComposeOp_CUPM(to_host, pobj, MatDenseCUPMRestoreArrayWrite_C(), nullptr, RestoreArrayC_<PETSC_MEMTYPE_DEVICE, PETSC_MEMORY_ACCESS_WRITE>);
    MatComposeOp_CUPM(to_host, pobj, MatDenseCUPMPlaceArray_C(), nullptr, PlaceArray);
    MatComposeOp_CUPM(to_host, pobj, MatDenseCUPMResetArray_C(), nullptr, ResetArray);
    MatComposeOp_CUPM(to_host, pobj, MatDenseCUPMReplaceArray_C(), nullptr, ReplaceArray);
    MatComposeOp_CUPM(to_host, pobj, MatProductSetFromOptions_seqaij_seqdensecupm_C(), nullptr, MatProductSetFromOptions_SeqAIJ_SeqDense);

    if (to_host) {
      B->offloadmask = PETSC_OFFLOAD_CPU;
    } else {
      Mat_SeqDenseCUPM *mcu;

      PetscCall(PetscNew(&mcu));
      B->spptr       = mcu;
      B->offloadmask = PETSC_OFFLOAD_UNALLOCATED; // REVIEW ME: why not offload host??
      PetscCall(BindToCPU(B, PETSC_FALSE));
    }

    MatSetOp_CUPM(to_host, B, bindtocpu, nullptr, BindToCPU);
    MatSetOp_CUPM(to_host, B, destroy, MatDestroy_SeqDense, Destroy);
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

// ==========================================================================================
// MatDense_Seq_CUPM - Public API
// ==========================================================================================

template <device::cupm::DeviceType T>
inline constexpr MatType MatDense_Seq_CUPM<T>::MATIMPLCUPM_() noexcept
{
  return MATSEQDENSECUPM();
}

template <device::cupm::DeviceType T>
inline constexpr typename MatDense_Seq_CUPM<T>::Mat_SeqDenseCUPM *MatDense_Seq_CUPM<T>::MatCUPMCast(Mat m) noexcept
{
  return static_cast<Mat_SeqDenseCUPM *>(m->spptr);
}

template <device::cupm::DeviceType T>
inline constexpr Mat_SeqDense *MatDense_Seq_CUPM<T>::MatIMPLCast_(Mat m) noexcept
{
  return static_cast<Mat_SeqDense *>(m->data);
}

template <device::cupm::DeviceType T>
inline constexpr const char *MatDense_Seq_CUPM<T>::MatConvert_seqdensecupm_seqdense_C() noexcept
{
  return T == device::cupm::DeviceType::CUDA ? "MatConvert_seqdensecuda_seqdense_C" : "MatConvert_seqdensehip_seqdense_C";
}

template <device::cupm::DeviceType T>
inline constexpr const char *MatDense_Seq_CUPM<T>::MatProductSetFromOptions_seqaij_seqdensecupm_C() noexcept
{
  return T == device::cupm::DeviceType::CUDA ? "MatProductSetFromOptions_seqaij_seqdensecuda_C" : "MatProductSetFromOptions_seqaij_seqdensehip_C";
}

// ==========================================================================================

// MatCreate_SeqDenseCUPM()
template <device::cupm::DeviceType T>
inline PetscErrorCode MatDense_Seq_CUPM<T>::Create(Mat A) noexcept
{
  PetscFunctionBegin;
  PetscCall(PetscDeviceInitialize(PETSC_DEVICE_CUPM()));
  PetscCall(MatCreate_SeqDense(A));
  PetscCall(Convert_SeqDense_SeqDenseCUPM(A, MATSEQDENSECUPM(), MAT_INPLACE_MATRIX, &A));
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <device::cupm::DeviceType T>
inline PetscErrorCode MatDense_Seq_CUPM<T>::Destroy(Mat A) noexcept
{
  PetscFunctionBegin;
  // prevent copying back data if we own the data pointer
  if (!MatIMPLCast(A)->user_alloc) A->offloadmask = PETSC_OFFLOAD_CPU;
  PetscCall(Convert_SeqDenseCUPM_SeqDense(A, MATSEQDENSE, MAT_INPLACE_MATRIX, &A));
  PetscCall(MatDestroy_SeqDense(A));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// obj->ops->setup()
template <device::cupm::DeviceType T>
inline PetscErrorCode MatDense_Seq_CUPM<T>::SetUp(Mat A) noexcept
{
  PetscFunctionBegin;
  PetscCall(PetscLayoutSetUp(A->rmap));
  PetscCall(PetscLayoutSetUp(A->cmap));
  if (!A->preallocated) {
    PetscDeviceContext dctx;

    PetscCall(GetHandles_(&dctx));
    PetscCall(SetPreallocation(A, dctx));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <device::cupm::DeviceType T>
inline PetscErrorCode MatDense_Seq_CUPM<T>::Reset(Mat A) noexcept
{
  PetscFunctionBegin;
  if (const auto mcu = MatCUPMCast(A)) {
    cupmStream_t stream;

    PetscCheck(!mcu->unplacedarray, PETSC_COMM_SELF, PETSC_ERR_ORDER, "MatDense%sResetArray() must be called first", cupmNAME());
    PetscCall(GetHandles_(&stream));
    if (!mcu->d_user_alloc) PetscCallCUPM(cupmFreeAsync(mcu->d_v, stream));
    PetscCallCUPM(cupmFreeAsync(mcu->d_fact_tau, stream));
    PetscCallCUPM(cupmFreeAsync(mcu->d_fact_ipiv, stream));
    PetscCallCUPM(cupmFreeAsync(mcu->d_fact_info, stream));
    PetscCallCUPM(cupmFreeAsync(mcu->d_fact_work, stream));
    PetscCall(VecDestroy(&mcu->workvec));
    PetscCall(PetscFree(A->spptr /* mcu */));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

// ==========================================================================================

template <device::cupm::DeviceType T>
inline PetscErrorCode MatDense_Seq_CUPM<T>::BindToCPU(Mat A, PetscBool to_host) noexcept
{
  const auto mimpl = MatIMPLCast(A);
  const auto pobj  = PetscObjectCast(A);

  PetscFunctionBegin;
  PetscCheck(!mimpl->vecinuse, PETSC_COMM_SELF, PETSC_ERR_ORDER, "Need to call MatDenseRestoreColumnVec() first");
  PetscCheck(!mimpl->matinuse, PETSC_COMM_SELF, PETSC_ERR_ORDER, "Need to call MatDenseRestoreSubMatrix() first");
  A->boundtocpu = to_host;
  PetscCall(PetscStrFreeAllocpy(to_host ? PETSCRANDER48 : PETSCDEVICERAND(), &A->defaultrandtype));
  if (to_host) {
    PetscDeviceContext dctx;

    // make sure we have an up-to-date copy on the CPU
    PetscCall(GetHandles_(&dctx));
    PetscCall(DeviceToHost_(A, dctx));
  } else {
    PetscBool iscupm;

    if (auto &cvec = mimpl->cvec) {
      PetscCall(PetscObjectTypeCompare(PetscObjectCast(cvec), VecSeq_CUPM::VECSEQCUPM(), &iscupm));
      if (!iscupm) PetscCall(VecDestroy(&cvec));
    }
    if (auto &cmat = mimpl->cmat) {
      PetscCall(PetscObjectTypeCompare(PetscObjectCast(cmat), MATSEQDENSECUPM(), &iscupm));
      if (!iscupm) PetscCall(MatDestroy(&cmat));
    }
  }

  // ============================================================
  // Composed ops
  // ============================================================
  MatComposeOp_CUPM(to_host, pobj, "MatDenseGetArray_C", MatDenseGetArray_SeqDense, GetArrayC_<PETSC_MEMTYPE_HOST, PETSC_MEMORY_ACCESS_READ_WRITE>);
  MatComposeOp_CUPM(to_host, pobj, "MatDenseGetArrayRead_C", MatDenseGetArray_SeqDense, GetArrayC_<PETSC_MEMTYPE_HOST, PETSC_MEMORY_ACCESS_READ>);
  MatComposeOp_CUPM(to_host, pobj, "MatDenseGetArrayWrite_C", MatDenseGetArray_SeqDense, GetArrayC_<PETSC_MEMTYPE_HOST, PETSC_MEMORY_ACCESS_WRITE>);
  MatComposeOp_CUPM(to_host, pobj, "MatDenseGetArrayAndMemType_C", nullptr, GetArrayAndMemTypeC_<PETSC_MEMORY_ACCESS_READ_WRITE>);
  MatComposeOp_CUPM(to_host, pobj, "MatDenseRestoreArrayAndMemType_C", nullptr, RestoreArrayAndMemTypeC_<PETSC_MEMORY_ACCESS_READ_WRITE>);
  MatComposeOp_CUPM(to_host, pobj, "MatDenseGetArrayReadAndMemType_C", nullptr, GetArrayAndMemTypeC_<PETSC_MEMORY_ACCESS_READ>);
  MatComposeOp_CUPM(to_host, pobj, "MatDenseRestoreArrayReadAndMemType_C", nullptr, RestoreArrayAndMemTypeC_<PETSC_MEMORY_ACCESS_READ>);
  MatComposeOp_CUPM(to_host, pobj, "MatDenseGetArrayWriteAndMemType_C", nullptr, GetArrayAndMemTypeC_<PETSC_MEMORY_ACCESS_WRITE>);
  MatComposeOp_CUPM(to_host, pobj, "MatDenseRestoreArrayWriteAndMemType_C", nullptr, RestoreArrayAndMemTypeC_<PETSC_MEMORY_ACCESS_WRITE>);
  MatComposeOp_CUPM(to_host, pobj, "MatDenseGetColumnVec_C", MatDenseGetColumnVec_SeqDense, GetColumnVec<PETSC_MEMORY_ACCESS_READ_WRITE>);
  MatComposeOp_CUPM(to_host, pobj, "MatDenseRestoreColumnVec_C", MatDenseRestoreColumnVec_SeqDense, RestoreColumnVec<PETSC_MEMORY_ACCESS_READ_WRITE>);
  MatComposeOp_CUPM(to_host, pobj, "MatDenseGetColumnVecRead_C", MatDenseGetColumnVecRead_SeqDense, GetColumnVec<PETSC_MEMORY_ACCESS_READ>);
  MatComposeOp_CUPM(to_host, pobj, "MatDenseRestoreColumnVecRead_C", MatDenseRestoreColumnVecRead_SeqDense, RestoreColumnVec<PETSC_MEMORY_ACCESS_READ>);
  MatComposeOp_CUPM(to_host, pobj, "MatDenseGetColumnVecWrite_C", MatDenseGetColumnVecWrite_SeqDense, GetColumnVec<PETSC_MEMORY_ACCESS_WRITE>);
  MatComposeOp_CUPM(to_host, pobj, "MatDenseRestoreColumnVecWrite_C", MatDenseRestoreColumnVecWrite_SeqDense, RestoreColumnVec<PETSC_MEMORY_ACCESS_WRITE>);
  MatComposeOp_CUPM(to_host, pobj, "MatDenseGetSubMatrix_C", MatDenseGetSubMatrix_SeqDense, GetSubMatrix);
  MatComposeOp_CUPM(to_host, pobj, "MatDenseRestoreSubMatrix_C", MatDenseRestoreSubMatrix_SeqDense, RestoreSubMatrix);
  MatComposeOp_CUPM(to_host, pobj, "MatQRFactor_C", MatQRFactor_SeqDense, SolveQR::Factor);
  // always the same
  PetscCall(PetscObjectComposeFunction(pobj, "MatDenseSetLDA_C", MatDenseSetLDA_SeqDense));

  // ============================================================
  // Function pointer ops
  // ============================================================
  MatSetOp_CUPM(to_host, A, duplicate, MatDuplicate_SeqDense, Duplicate);
  MatSetOp_CUPM(to_host, A, mult, MatMult_SeqDense, [](Mat A, Vec xx, Vec yy) { return MatMultAdd_Dispatch_</* transpose */ false>(A, xx, nullptr, yy); });
  MatSetOp_CUPM(to_host, A, multtranspose, MatMultTranspose_SeqDense, [](Mat A, Vec xx, Vec yy) { return MatMultAdd_Dispatch_</* transpose */ true>(A, xx, nullptr, yy); });
  MatSetOp_CUPM(to_host, A, multadd, MatMultAdd_SeqDense, MatMultAdd_Dispatch_</* transpose */ false>);
  MatSetOp_CUPM(to_host, A, multtransposeadd, MatMultTransposeAdd_SeqDense, MatMultAdd_Dispatch_</* transpose */ true>);
  MatSetOp_CUPM(to_host, A, matmultnumeric, MatMatMultNumeric_SeqDense_SeqDense, MatMatMult_Numeric_Dispatch</* transpose_A */ false, /* transpose_B */ false>);
  MatSetOp_CUPM(to_host, A, mattransposemultnumeric, MatMatTransposeMultNumeric_SeqDense_SeqDense, MatMatMult_Numeric_Dispatch</* transpose_A */ false, /* transpose_B */ true>);
  MatSetOp_CUPM(to_host, A, transposematmultnumeric, MatTransposeMatMultNumeric_SeqDense_SeqDense, MatMatMult_Numeric_Dispatch</* transpose_A */ true, /* transpose_B */ false>);
  MatSetOp_CUPM(to_host, A, axpy, MatAXPY_SeqDense, AXPY);
  MatSetOp_CUPM(to_host, A, choleskyfactor, MatCholeskyFactor_SeqDense, SolveCholesky::Factor);
  MatSetOp_CUPM(to_host, A, lufactor, MatLUFactor_SeqDense, SolveLU::Factor);
  MatSetOp_CUPM(to_host, A, getcolumnvector, MatGetColumnVector_SeqDense, GetColumnVector);
  MatSetOp_CUPM(to_host, A, scale, MatScale_SeqDense, Scale);
  MatSetOp_CUPM(to_host, A, shift, MatShift_SeqDense, Shift);
  MatSetOp_CUPM(to_host, A, copy, MatCopy_SeqDense, Copy);
  MatSetOp_CUPM(to_host, A, zeroentries, MatZeroEntries_SeqDense, ZeroEntries);
  MatSetOp_CUPM(to_host, A, setup, MatSetUp_SeqDense, SetUp);
  MatSetOp_CUPM(to_host, A, setrandom, MatSetRandom_SeqDense, SetRandom);
  // seemingly always the same
  A->ops->productsetfromoptions = MatProductSetFromOptions_SeqDense;

  if (const auto cmat = mimpl->cmat) PetscCall(MatBindToCPU(cmat, to_host));
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <device::cupm::DeviceType T>
inline PetscErrorCode MatDense_Seq_CUPM<T>::Convert_SeqDenseCUPM_SeqDense(Mat M, MatType type, MatReuse reuse, Mat *newmat) noexcept
{
  PetscFunctionBegin;
  PetscCall(Convert_Dispatch_</* to host */ true>(M, type, reuse, newmat));
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <device::cupm::DeviceType T>
inline PetscErrorCode MatDense_Seq_CUPM<T>::Convert_SeqDense_SeqDenseCUPM(Mat M, MatType type, MatReuse reuse, Mat *newmat) noexcept
{
  PetscFunctionBegin;
  PetscCall(Convert_Dispatch_</* to host */ false>(M, type, reuse, newmat));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// ==========================================================================================

template <device::cupm::DeviceType T>
template <PetscMemType mtype, PetscMemoryAccessMode access>
inline PetscErrorCode MatDense_Seq_CUPM<T>::GetArray(Mat m, PetscScalar **array, PetscDeviceContext dctx) noexcept
{
  constexpr auto hostmem     = PetscMemTypeHost(mtype);
  constexpr auto read_access = PetscMemoryAccessRead(access);

  PetscFunctionBegin;
  static_assert((mtype == PETSC_MEMTYPE_HOST) || (mtype == PETSC_MEMTYPE_DEVICE), "");
  PetscCall(PetscDeviceContextGetOptionalNullContext_Internal(&dctx));
  if (hostmem) {
    if (read_access) {
      PetscCall(DeviceToHost_(m, dctx));
    } else if (!MatIMPLCast(m)->v) {
      // MatCreateSeqDenseCUPM may not allocate CPU memory. Allocate if needed
      PetscCall(MatSeqDenseSetPreallocation(m, nullptr));
    }
    *array = MatIMPLCast(m)->v;
  } else {
    if (read_access) {
      PetscCall(HostToDevice_(m, dctx));
    } else if (!MatCUPMCast(m)->d_v) {
      // write-only
      PetscCall(SetPreallocation(m, dctx, nullptr));
    }
    *array = MatCUPMCast(m)->d_v;
  }
  if (PetscMemoryAccessWrite(access)) {
    m->offloadmask = hostmem ? PETSC_OFFLOAD_CPU : PETSC_OFFLOAD_GPU;
    PetscCall(PetscObjectStateIncrease(PetscObjectCast(m)));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <device::cupm::DeviceType T>
template <PetscMemType mtype, PetscMemoryAccessMode access>
inline PetscErrorCode MatDense_Seq_CUPM<T>::RestoreArray(Mat m, PetscScalar **array, PetscDeviceContext) noexcept
{
  PetscFunctionBegin;
  static_assert((mtype == PETSC_MEMTYPE_HOST) || (mtype == PETSC_MEMTYPE_DEVICE), "");
  if (PetscMemoryAccessWrite(access)) {
    // WRITE or READ_WRITE
    m->offloadmask = PetscMemTypeHost(mtype) ? PETSC_OFFLOAD_CPU : PETSC_OFFLOAD_GPU;
    PetscCall(PetscObjectStateIncrease(PetscObjectCast(m)));
  }
  if (array) {
    PetscCall(CheckPointerMatchesMemType_(*array, mtype));
    *array = nullptr;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <device::cupm::DeviceType T>
template <PetscMemoryAccessMode access>
inline PetscErrorCode MatDense_Seq_CUPM<T>::GetArrayAndMemType(Mat m, PetscScalar **array, PetscMemType *mtype, PetscDeviceContext dctx) noexcept
{
  PetscFunctionBegin;
  PetscCall(GetArray<PETSC_MEMTYPE_DEVICE, access>(m, array, dctx));
  if (mtype) *mtype = PETSC_MEMTYPE_CUPM();
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <device::cupm::DeviceType T>
template <PetscMemoryAccessMode access>
inline PetscErrorCode MatDense_Seq_CUPM<T>::RestoreArrayAndMemType(Mat m, PetscScalar **array, PetscDeviceContext dctx) noexcept
{
  PetscFunctionBegin;
  PetscCall(RestoreArray<PETSC_MEMTYPE_DEVICE, access>(m, array, dctx));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// ==========================================================================================

template <device::cupm::DeviceType T>
inline PetscErrorCode MatDense_Seq_CUPM<T>::PlaceArray(Mat A, const PetscScalar *array) noexcept
{
  const auto mimpl = MatIMPLCast(A);
  const auto mcu   = MatCUPMCast(A);

  PetscFunctionBegin;
  PetscCheck(!mimpl->vecinuse, PETSC_COMM_SELF, PETSC_ERR_ORDER, "Need to call MatDenseRestoreColumnVec() first");
  PetscCheck(!mimpl->matinuse, PETSC_COMM_SELF, PETSC_ERR_ORDER, "Need to call MatDenseRestoreSubMatrix() first");
  PetscCheck(!mcu->unplacedarray, PETSC_COMM_SELF, PETSC_ERR_ORDER, "MatDense%sResetArray() must be called first", cupmNAME());
  if (mimpl->v) {
    PetscDeviceContext dctx;

    PetscCall(GetHandles_(&dctx));
    PetscCall(HostToDevice_(A, dctx));
  }
  mcu->unplacedarray         = util::exchange(mcu->d_v, const_cast<PetscScalar *>(array));
  mcu->d_unplaced_user_alloc = util::exchange(mcu->d_user_alloc, PETSC_TRUE);
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <device::cupm::DeviceType T>
inline PetscErrorCode MatDense_Seq_CUPM<T>::ReplaceArray(Mat A, const PetscScalar *array) noexcept
{
  const auto mimpl = MatIMPLCast(A);
  const auto mcu   = MatCUPMCast(A);

  PetscFunctionBegin;
  PetscCheck(!mimpl->vecinuse, PETSC_COMM_SELF, PETSC_ERR_ORDER, "Need to call MatDenseRestoreColumnVec() first");
  PetscCheck(!mimpl->matinuse, PETSC_COMM_SELF, PETSC_ERR_ORDER, "Need to call MatDenseRestoreSubMatrix() first");
  PetscCheck(!mcu->unplacedarray, PETSC_COMM_SELF, PETSC_ERR_ORDER, "MatDense%sResetArray() must be called first", cupmNAME());
  if (!mcu->d_user_alloc) {
    cupmStream_t stream;

    PetscCall(GetHandles_(&stream));
    PetscCallCUPM(cupmFreeAsync(mcu->d_v, stream));
  }
  mcu->d_v          = const_cast<PetscScalar *>(array);
  mcu->d_user_alloc = PETSC_FALSE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <device::cupm::DeviceType T>
inline PetscErrorCode MatDense_Seq_CUPM<T>::ResetArray(Mat A) noexcept
{
  const auto mimpl = MatIMPLCast(A);
  const auto mcu   = MatCUPMCast(A);

  PetscFunctionBegin;
  PetscCheck(!mimpl->vecinuse, PETSC_COMM_SELF, PETSC_ERR_ORDER, "Need to call MatDenseRestoreColumnVec() first");
  PetscCheck(!mimpl->matinuse, PETSC_COMM_SELF, PETSC_ERR_ORDER, "Need to call MatDenseRestoreSubMatrix() first");
  if (mimpl->v) {
    PetscDeviceContext dctx;

    PetscCall(GetHandles_(&dctx));
    PetscCall(HostToDevice_(A, dctx));
  }
  mcu->d_v          = util::exchange(mcu->unplacedarray, nullptr);
  mcu->d_user_alloc = mcu->d_unplaced_user_alloc;
  PetscFunctionReturn(PETSC_SUCCESS);
}

// ==========================================================================================

template <device::cupm::DeviceType T>
template <bool transpose_A, bool transpose_B>
inline PetscErrorCode MatDense_Seq_CUPM<T>::MatMatMult_Numeric_Dispatch(Mat A, Mat B, Mat C) noexcept
{
  cupmBlasInt_t      m, n, k;
  PetscBool          Aiscupm, Biscupm;
  PetscDeviceContext dctx;
  cupmBlasHandle_t   handle;

  PetscFunctionBegin;
  PetscCall(PetscCUPMBlasIntCast(C->rmap->n, &m));
  PetscCall(PetscCUPMBlasIntCast(C->cmap->n, &n));
  PetscCall(PetscCUPMBlasIntCast(transpose_A ? A->rmap->n : A->cmap->n, &k));
  if (!m || !n || !k) PetscFunctionReturn(PETSC_SUCCESS);

  // we may end up with SEQDENSE as one of the arguments
  // REVIEW ME: how? and why is it not B and C????????
  PetscCall(PetscObjectTypeCompare(PetscObjectCast(A), MATSEQDENSECUPM(), &Aiscupm));
  PetscCall(PetscObjectTypeCompare(PetscObjectCast(B), MATSEQDENSECUPM(), &Biscupm));
  if (!Aiscupm) PetscCall(MatConvert(A, MATSEQDENSECUPM(), MAT_INPLACE_MATRIX, &A));
  if (!Biscupm) PetscCall(MatConvert(B, MATSEQDENSECUPM(), MAT_INPLACE_MATRIX, &B));
  PetscCall(PetscInfo(C, "Matrix-Matrix product %" PetscBLASInt_FMT " x %" PetscBLASInt_FMT " x %" PetscBLASInt_FMT " on backend\n", m, k, n));
  PetscCall(GetHandles_(&dctx, &handle));

  PetscCall(PetscLogGpuTimeBegin());
  {
    const auto one  = cupmScalarCast(1.0);
    const auto zero = cupmScalarCast(0.0);
    const auto da   = DeviceArrayRead(dctx, A);
    const auto db   = DeviceArrayRead(dctx, B);
    const auto dc   = DeviceArrayWrite(dctx, C);
    PetscInt   alda, blda, clda;

    PetscCall(MatDenseGetLDA(A, &alda));
    PetscCall(MatDenseGetLDA(B, &blda));
    PetscCall(MatDenseGetLDA(C, &clda));
    PetscCallCUPMBLAS(cupmBlasXgemm(handle, transpose_A ? CUPMBLAS_OP_T : CUPMBLAS_OP_N, transpose_B ? CUPMBLAS_OP_T : CUPMBLAS_OP_N, m, n, k, &one, da.cupmdata(), alda, db.cupmdata(), blda, &zero, dc.cupmdata(), clda));
  }
  PetscCall(PetscLogGpuTimeEnd());

  PetscCall(PetscLogGpuFlops(1.0 * m * n * k + 1.0 * m * n * (k - 1)));
  if (!Aiscupm) PetscCall(MatConvert(A, MATSEQDENSE, MAT_INPLACE_MATRIX, &A));
  if (!Biscupm) PetscCall(MatConvert(B, MATSEQDENSE, MAT_INPLACE_MATRIX, &B));
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <device::cupm::DeviceType T>
inline PetscErrorCode MatDense_Seq_CUPM<T>::Copy(Mat A, Mat B, MatStructure str) noexcept
{
  const auto m = A->rmap->n;
  const auto n = A->cmap->n;

  PetscFunctionBegin;
  PetscAssert(m == B->rmap->n && n == B->cmap->n, PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "size(B) != size(A)");
  // The two matrices must have the same copy implementation to be eligible for fast copy
  if (A->ops->copy == B->ops->copy) {
    PetscDeviceContext dctx;
    cupmStream_t       stream;

    PetscCall(GetHandles_(&dctx, &stream));
    PetscCall(PetscLogGpuTimeBegin());
    {
      const auto va = DeviceArrayRead(dctx, A);
      const auto vb = DeviceArrayWrite(dctx, B);
      // order is important, DeviceArrayRead/Write() might call SetPreallocation() which sets
      // lda!
      const auto lda_a = MatIMPLCast(A)->lda;
      const auto lda_b = MatIMPLCast(B)->lda;

      if (lda_a > m || lda_b > m) {
        PetscAssert(lda_b > 0, PETSC_COMM_SELF, PETSC_ERR_PLIB, "B lda (%" PetscBLASInt_FMT ") must be > 0 at this point, this indicates Mat%sSetPreallocation() was not called when it should have been!", lda_b, cupmNAME());
        PetscAssert(lda_a > 0, PETSC_COMM_SELF, PETSC_ERR_PLIB, "A lda (%" PetscBLASInt_FMT ") must be > 0 at this point, this indicates Mat%sSetPreallocation() was not called when it should have been!", lda_a, cupmNAME());
        PetscCall(PetscCUPMMemcpy2DAsync(vb.data(), lda_b, va.data(), lda_a, m, n, cupmMemcpyDeviceToDevice, stream));
      } else {
        PetscCall(PetscCUPMMemcpyAsync(vb.data(), va.data(), m * n, cupmMemcpyDeviceToDevice, stream));
      }
    }
    PetscCall(PetscLogGpuTimeEnd());
  } else {
    PetscCall(MatCopy_Basic(A, B, str));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <device::cupm::DeviceType T>
inline PetscErrorCode MatDense_Seq_CUPM<T>::ZeroEntries(Mat m) noexcept
{
  PetscDeviceContext dctx;
  cupmStream_t       stream;

  PetscFunctionBegin;
  PetscCall(GetHandles_(&dctx, &stream));
  PetscCall(PetscLogGpuTimeBegin());
  {
    const auto va  = DeviceArrayWrite(dctx, m);
    const auto lda = MatIMPLCast(m)->lda;
    const auto ma  = m->rmap->n;
    const auto na  = m->cmap->n;

    if (lda > ma) {
      PetscCall(PetscCUPMMemset2DAsync(va.data(), lda, 0, ma, na, stream));
    } else {
      PetscCall(PetscCUPMMemsetAsync(va.data(), 0, ma * na, stream));
    }
  }
  PetscCall(PetscLogGpuTimeEnd());
  PetscFunctionReturn(PETSC_SUCCESS);
}

namespace detail
{

// ==========================================================================================
// SubMatIndexFunctor
//
// Iterator which permutes a linear index range into matrix indices for am nrows x ncols
// submat with leading dimension lda. Essentially SubMatIndexFunctor(i) returns the index for
// the i'th sequential entry in the matrix.
// ==========================================================================================
template <typename T>
struct SubMatIndexFunctor {
  PETSC_HOSTDEVICE_INLINE_DECL T operator()(T x) const noexcept { return ((x / nrows) * lda) + (x % nrows); }

  PetscInt nrows;
  PetscInt ncols;
  PetscInt lda;
};

template <typename Iterator>
struct SubMatrixIterator : MatrixIteratorBase<Iterator, SubMatIndexFunctor<typename thrust::iterator_difference<Iterator>::type>> {
  using base_type = MatrixIteratorBase<Iterator, SubMatIndexFunctor<typename thrust::iterator_difference<Iterator>::type>>;

  using iterator = typename base_type::iterator;

  constexpr SubMatrixIterator(Iterator first, Iterator last, PetscInt nrows, PetscInt ncols, PetscInt lda) noexcept :
    base_type{
      std::move(first), std::move(last), {nrows, ncols, lda}
  }
  {
  }

  PETSC_NODISCARD iterator end() const noexcept { return this->begin() + (this->func.nrows * this->func.ncols); }
};

namespace
{

template <typename T>
PETSC_NODISCARD inline SubMatrixIterator<typename thrust::device_vector<T>::iterator> make_submat_iterator(PetscInt rstart, PetscInt rend, PetscInt cstart, PetscInt cend, PetscInt lda, T *ptr) noexcept
{
  const auto nrows = rend - rstart;
  const auto ncols = cend - cstart;
  const auto dptr  = thrust::device_pointer_cast(ptr);

  return {dptr + (rstart * lda) + cstart, dptr + ((rstart + nrows) * lda) + cstart, nrows, ncols, lda};
}

} // namespace

} // namespace detail

template <device::cupm::DeviceType T>
inline PetscErrorCode MatDense_Seq_CUPM<T>::Scale(Mat A, PetscScalar alpha) noexcept
{
  const auto         m = A->rmap->n;
  const auto         n = A->cmap->n;
  const auto         N = m * n;
  PetscDeviceContext dctx;

  PetscFunctionBegin;
  PetscCall(PetscInfo(A, "Performing Scale %" PetscInt_FMT " x %" PetscInt_FMT " on backend\n", m, n));
  PetscCall(GetHandles_(&dctx));
  {
    const auto da  = DeviceArrayReadWrite(dctx, A);
    const auto lda = MatIMPLCast(A)->lda;

    if (lda > m) {
      cupmStream_t stream;

      PetscCall(GetHandlesFrom_(dctx, &stream));
      // clang-format off
      PetscCallThrust(
        const auto sub_mat = detail::make_submat_iterator(0, m, 0, n, lda, da.data());

        THRUST_CALL(
          thrust::transform,
          stream,
          sub_mat.begin(), sub_mat.end(), sub_mat.begin(),
          device::cupm::functors::make_times_equals(alpha)
        )
      );
      // clang-format on
    } else {
      const auto       cu_alpha = cupmScalarCast(alpha);
      cupmBlasHandle_t handle;

      PetscCall(GetHandlesFrom_(dctx, &handle));
      PetscCall(PetscLogGpuTimeBegin());
      PetscCallCUPMBLAS(cupmBlasXscal(handle, N, &cu_alpha, da.cupmdata(), 1));
      PetscCall(PetscLogGpuTimeEnd());
    }
  }
  PetscCall(PetscLogGpuFlops(N));
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <device::cupm::DeviceType T>
inline PetscErrorCode MatDense_Seq_CUPM<T>::Shift(Mat A, PetscScalar alpha) noexcept
{
  const auto         m = A->rmap->n;
  const auto         n = A->cmap->n;
  PetscDeviceContext dctx;

  PetscFunctionBegin;
  PetscCall(GetHandles_(&dctx));
  PetscCall(PetscInfo(A, "Performing Shift %" PetscInt_FMT " x %" PetscInt_FMT " on backend\n", m, n));
  PetscCall(DiagonalUnaryTransform(A, 0, m, n, dctx, device::cupm::functors::make_plus_equals(alpha)));
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <device::cupm::DeviceType T>
inline PetscErrorCode MatDense_Seq_CUPM<T>::AXPY(Mat Y, PetscScalar alpha, Mat X, MatStructure) noexcept
{
  const auto         m_x = X->rmap->n, m_y = Y->rmap->n;
  const auto         n_x = X->cmap->n, n_y = Y->cmap->n;
  const auto         N = m_x * n_x;
  PetscDeviceContext dctx;

  PetscFunctionBegin;
  if (!m_x || !n_x || alpha == (PetscScalar)0.0) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(PetscInfo(Y, "Performing AXPY %" PetscInt_FMT " x %" PetscInt_FMT " on backend\n", m_y, n_y));
  PetscCall(GetHandles_(&dctx));
  {
    const auto dx    = DeviceArrayRead(dctx, X);
    const auto dy    = DeviceArrayReadWrite(dctx, Y);
    const auto lda_x = MatIMPLCast(X)->lda;
    const auto lda_y = MatIMPLCast(Y)->lda;

    if (lda_x > m_x || lda_y > m_x) {
      cupmStream_t stream;

      PetscCall(GetHandlesFrom_(dctx, &stream));
      // clang-format off
      PetscCallThrust(
        const auto sub_mat_y = detail::make_submat_iterator(0, m_y, 0, n_y, lda_y, dy.data());
        const auto sub_mat_x = detail::make_submat_iterator(0, m_x, 0, n_x, lda_x, dx.data());

        THRUST_CALL(
          thrust::transform,
          stream,
          sub_mat_x.begin(), sub_mat_x.end(), sub_mat_y.begin(), sub_mat_y.begin(),
          device::cupm::functors::make_axpy(alpha)
        );
      );
      // clang-format on
    } else {
      const auto       cu_alpha = cupmScalarCast(alpha);
      cupmBlasHandle_t handle;

      PetscCall(GetHandlesFrom_(dctx, &handle));
      PetscCall(PetscLogGpuTimeBegin());
      PetscCallCUPMBLAS(cupmBlasXaxpy(handle, N, &cu_alpha, dx.cupmdata(), 1, dy.cupmdata(), 1));
      PetscCall(PetscLogGpuTimeEnd());
    }
  }
  PetscCall(PetscLogGpuFlops(PetscMax(2 * N - 1, 0)));
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <device::cupm::DeviceType T>
inline PetscErrorCode MatDense_Seq_CUPM<T>::Duplicate(Mat A, MatDuplicateOption opt, Mat *B) noexcept
{
  const auto         hopt = (opt == MAT_COPY_VALUES && A->offloadmask != PETSC_OFFLOAD_CPU) ? MAT_DO_NOT_COPY_VALUES : opt;
  PetscDeviceContext dctx;

  PetscFunctionBegin;
  PetscCall(GetHandles_(&dctx));
  // do not call SetPreallocation() yet, we call it afterwards??
  PetscCall(MatCreateSeqDenseCUPM<T>(PetscObjectComm(PetscObjectCast(A)), A->rmap->n, A->cmap->n, nullptr, B, dctx, /* preallocate */ false));
  PetscCall(MatDuplicateNoCreate_SeqDense(*B, A, hopt));
  if (opt == MAT_COPY_VALUES && hopt != MAT_COPY_VALUES) PetscCall(Copy(A, *B, SAME_NONZERO_PATTERN));
  // allocate memory if needed
  if (opt != MAT_COPY_VALUES && !MatCUPMCast(*B)->d_v) PetscCall(SetPreallocation(*B, dctx));
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <device::cupm::DeviceType T>
inline PetscErrorCode MatDense_Seq_CUPM<T>::SetRandom(Mat A, PetscRandom rng) noexcept
{
  PetscBool device;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare(PetscObjectCast(rng), PETSCDEVICERAND(), &device));
  if (device) {
    const auto         m = A->rmap->n;
    const auto         n = A->cmap->n;
    PetscDeviceContext dctx;

    PetscCall(GetHandles_(&dctx));
    {
      const auto a = DeviceArrayWrite(dctx, A);
      PetscInt   lda;

      PetscCall(MatDenseGetLDA(A, &lda));
      if (lda > m) {
        for (PetscInt i = 0; i < n; i++) PetscCall(PetscRandomGetValues(rng, m, a.data() + i * lda));
      } else {
        PetscInt mn;

        PetscCall(PetscIntMultError(m, n, &mn));
        PetscCall(PetscRandomGetValues(rng, mn, a));
      }
    }
  } else {
    PetscCall(MatSetRandom_SeqDense(A, rng));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

// ==========================================================================================

template <device::cupm::DeviceType T>
inline PetscErrorCode MatDense_Seq_CUPM<T>::GetColumnVector(Mat A, Vec v, PetscInt col) noexcept
{
  const auto         offloadmask = A->offloadmask;
  const auto         n           = A->rmap->n;
  const auto         col_offset  = [&](const PetscScalar *ptr) { return ptr + col * MatIMPLCast(A)->lda; };
  PetscBool          viscupm;
  PetscDeviceContext dctx;
  cupmStream_t       stream;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompareAny(PetscObjectCast(v), &viscupm, VecSeq_CUPM::VECSEQCUPM(), VecSeq_CUPM::VECMPICUPM(), VecSeq_CUPM::VECCUPM(), ""));
  PetscCall(GetHandles_(&dctx, &stream));
  if (viscupm && !v->boundtocpu) {
    const auto x = VecSeq_CUPM::DeviceArrayWrite(dctx, v);

    // update device data
    if (PetscOffloadDevice(offloadmask)) {
      PetscCall(PetscCUPMMemcpyAsync(x.data(), col_offset(DeviceArrayRead(dctx, A)), n, cupmMemcpyDeviceToDevice, stream));
    } else {
      PetscCall(PetscCUPMMemcpyAsync(x.data(), col_offset(HostArrayRead(dctx, A)), n, cupmMemcpyHostToDevice, stream));
    }
  } else {
    PetscScalar *x;

    // update host data
    PetscCall(VecGetArrayWrite(v, &x));
    if (PetscOffloadUnallocated(offloadmask) || PetscOffloadHost(offloadmask)) {
      PetscCall(PetscArraycpy(x, col_offset(HostArrayRead(dctx, A)), n));
    } else if (PetscOffloadDevice(offloadmask)) {
      PetscCall(PetscCUPMMemcpyAsync(x, col_offset(DeviceArrayRead(dctx, A)), n, cupmMemcpyDeviceToHost, stream));
    }
    PetscCall(VecRestoreArrayWrite(v, &x));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <device::cupm::DeviceType T>
template <PetscMemoryAccessMode access>
inline PetscErrorCode MatDense_Seq_CUPM<T>::GetColumnVec(Mat A, PetscInt col, Vec *v) noexcept
{
  using namespace vec::cupm;
  const auto         mimpl = MatIMPLCast(A);
  PetscDeviceContext dctx;

  PetscFunctionBegin;
  PetscCheck(!mimpl->vecinuse, PETSC_COMM_SELF, PETSC_ERR_ORDER, "Need to call MatDenseRestoreColumnVec() first");
  PetscCheck(!mimpl->matinuse, PETSC_COMM_SELF, PETSC_ERR_ORDER, "Need to call MatDenseRestoreSubMatrix() first");
  mimpl->vecinuse = col + 1;
  PetscCall(GetHandles_(&dctx));
  PetscCall(GetArray<PETSC_MEMTYPE_DEVICE, access>(A, const_cast<PetscScalar **>(&mimpl->ptrinuse), dctx));
  if (!mimpl->cvec) {
    // we pass the data of A, to prevent allocating needless GPU memory the first time
    // VecCUPMPlaceArray is called
    PetscCall(VecCreateSeqCUPMWithArraysAsync<T>(PetscObjectComm(PetscObjectCast(A)), A->rmap->bs, A->rmap->n, nullptr, mimpl->ptrinuse, &mimpl->cvec));
  }
  PetscCall(VecCUPMPlaceArrayAsync<T>(mimpl->cvec, mimpl->ptrinuse + static_cast<std::size_t>(col) * static_cast<std::size_t>(mimpl->lda)));
  if (access == PETSC_MEMORY_ACCESS_READ) PetscCall(VecLockReadPush(mimpl->cvec));
  *v = mimpl->cvec;
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <device::cupm::DeviceType T>
template <PetscMemoryAccessMode access>
inline PetscErrorCode MatDense_Seq_CUPM<T>::RestoreColumnVec(Mat A, PetscInt, Vec *v) noexcept
{
  using namespace vec::cupm;
  const auto         mimpl = MatIMPLCast(A);
  const auto         cvec  = mimpl->cvec;
  PetscDeviceContext dctx;

  PetscFunctionBegin;
  PetscCheck(mimpl->vecinuse, PETSC_COMM_SELF, PETSC_ERR_ORDER, "Need to call MatDenseGetColumnVec() first");
  PetscCheck(cvec, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Missing internal column vector");
  mimpl->vecinuse = 0;
  if (access == PETSC_MEMORY_ACCESS_READ) PetscCall(VecLockReadPop(cvec));
  PetscCall(VecCUPMResetArrayAsync<T>(cvec));
  PetscCall(GetHandles_(&dctx));
  PetscCall(RestoreArray<PETSC_MEMTYPE_DEVICE, access>(A, const_cast<PetscScalar **>(&mimpl->ptrinuse), dctx));
  if (v) *v = nullptr;
  PetscFunctionReturn(PETSC_SUCCESS);
}

// ==========================================================================================

template <device::cupm::DeviceType T>
inline PetscErrorCode MatDense_Seq_CUPM<T>::GetFactor(Mat A, MatFactorType ftype, Mat *fact_out) noexcept
{
  Mat                fact = nullptr;
  PetscDeviceContext dctx;

  PetscFunctionBegin;
  PetscCall(GetHandles_(&dctx));
  PetscCall(MatCreateSeqDenseCUPM<T>(PetscObjectComm(PetscObjectCast(A)), A->rmap->n, A->cmap->n, nullptr, &fact, dctx, /* preallocate */ false));
  fact->factortype = ftype;
  switch (ftype) {
  case MAT_FACTOR_LU:
  case MAT_FACTOR_ILU: // fall-through
    fact->ops->lufactorsymbolic  = MatLUFactorSymbolic_SeqDense;
    fact->ops->ilufactorsymbolic = MatLUFactorSymbolic_SeqDense;
    break;
  case MAT_FACTOR_CHOLESKY:
  case MAT_FACTOR_ICC: // fall-through
    fact->ops->choleskyfactorsymbolic = MatCholeskyFactorSymbolic_SeqDense;
    break;
  case MAT_FACTOR_QR: {
    const auto pobj = PetscObjectCast(fact);

    PetscCall(PetscObjectComposeFunction(pobj, "MatQRFactor_C", MatQRFactor_SeqDense));
    PetscCall(PetscObjectComposeFunction(pobj, "MatQRFactorSymbolic_C", MatQRFactorSymbolic_SeqDense));
  } break;
  case MAT_FACTOR_NONE:
  case MAT_FACTOR_ILUDT:     // fall-through
  case MAT_FACTOR_NUM_TYPES: // fall-through
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "MatFactorType %s not supported", MatFactorTypes[ftype]);
  }
  PetscCall(PetscStrFreeAllocpy(MATSOLVERCUPM(), &fact->solvertype));
  PetscCall(PetscStrallocpy(MATORDERINGEXTERNAL, const_cast<char **>(fact->preferredordering) + MAT_FACTOR_LU));
  PetscCall(PetscStrallocpy(MATORDERINGEXTERNAL, const_cast<char **>(fact->preferredordering) + MAT_FACTOR_ILU));
  PetscCall(PetscStrallocpy(MATORDERINGEXTERNAL, const_cast<char **>(fact->preferredordering) + MAT_FACTOR_CHOLESKY));
  PetscCall(PetscStrallocpy(MATORDERINGEXTERNAL, const_cast<char **>(fact->preferredordering) + MAT_FACTOR_ICC));
  *fact_out = fact;
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <device::cupm::DeviceType T>
inline PetscErrorCode MatDense_Seq_CUPM<T>::InvertFactors(Mat A) noexcept
{
  const auto         mimpl = MatIMPLCast(A);
  const auto         mcu   = MatCUPMCast(A);
  const auto         n     = static_cast<cupmBlasInt_t>(A->cmap->n);
  cupmSolverHandle_t handle;
  PetscDeviceContext dctx;
  cupmStream_t       stream;

  PetscFunctionBegin;
  #if PetscDefined(HAVE_CUDA) && PetscDefined(USING_NVCC)
  // HIP appears to have this by default??
  PetscCheck(PETSC_PKG_CUDA_VERSION_GE(10, 1, 0), PETSC_COMM_SELF, PETSC_ERR_SUP, "Upgrade to CUDA version 10.1.0 or higher");
  #endif
  if (!n || !A->rmap->n) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCheck(A->factortype == MAT_FACTOR_CHOLESKY, PETSC_COMM_SELF, PETSC_ERR_LIB, "Factor type %s not implemented", MatFactorTypes[A->factortype]);
  // spd
  PetscCheck(!mcu->d_fact_ipiv, PETSC_COMM_SELF, PETSC_ERR_LIB, "%sDnsytri not implemented", cupmSolverName());

  PetscCall(GetHandles_(&dctx, &handle, &stream));
  {
    const auto    da  = DeviceArrayReadWrite(dctx, A);
    const auto    lda = static_cast<cupmBlasInt_t>(mimpl->lda);
    cupmBlasInt_t il;

    PetscCallCUPMSOLVER(cupmSolverXpotri_bufferSize(handle, CUPMSOLVER_FILL_MODE_LOWER, n, da.cupmdata(), lda, &il));
    if (il > mcu->d_fact_lwork) {
      mcu->d_fact_lwork = il;
      PetscCallCUPM(cupmFreeAsync(mcu->d_fact_work, stream));
      PetscCall(PetscCUPMMallocAsync(&mcu->d_fact_work, il, stream));
    }
    PetscCall(PetscLogGpuTimeBegin());
    PetscCallCUPMSOLVER(cupmSolverXpotri(handle, CUPMSOLVER_FILL_MODE_LOWER, n, da.cupmdata(), lda, mcu->d_fact_work, mcu->d_fact_lwork, mcu->d_fact_info));
    PetscCall(PetscLogGpuTimeEnd());
  }
  PetscCall(CheckCUPMSolverInfo_(mcu->d_fact_info, stream));
  // TODO (write cuda kernel)
  PetscCall(MatSeqDenseSymmetrize_Private(A, PETSC_TRUE));
  PetscCall(PetscLogGpuFlops(1.0 * n * n * n / 3.0));

  A->ops->solve          = nullptr;
  A->ops->solvetranspose = nullptr;
  A->ops->matsolve       = nullptr;
  A->factortype          = MAT_FACTOR_NONE;

  PetscCall(PetscFree(A->solvertype));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// ==========================================================================================

template <device::cupm::DeviceType T>
inline PetscErrorCode MatDense_Seq_CUPM<T>::GetSubMatrix(Mat A, PetscInt rbegin, PetscInt rend, PetscInt cbegin, PetscInt cend, Mat *mat) noexcept
{
  const auto         mimpl        = MatIMPLCast(A);
  const auto         array_offset = [&](PetscScalar *ptr) { return ptr + rbegin + static_cast<std::size_t>(cbegin) * mimpl->lda; };
  const auto         n            = rend - rbegin;
  const auto         m            = cend - cbegin;
  auto              &cmat         = mimpl->cmat;
  PetscDeviceContext dctx;

  PetscFunctionBegin;
  PetscCheck(!mimpl->vecinuse, PETSC_COMM_SELF, PETSC_ERR_ORDER, "Need to call MatDenseRestoreColumnVec() first");
  PetscCheck(!mimpl->matinuse, PETSC_COMM_SELF, PETSC_ERR_ORDER, "Need to call MatDenseRestoreSubMatrix() first");
  mimpl->matinuse = cbegin + 1;

  PetscCall(GetHandles_(&dctx));
  PetscCall(HostToDevice_(A, dctx));

  if (cmat && ((m != cmat->cmap->N) || (n != cmat->rmap->N))) PetscCall(MatDestroy(&cmat));
  {
    const auto device_array = array_offset(MatCUPMCast(A)->d_v);

    if (cmat) {
      PetscCall(PlaceArray(cmat, device_array));
    } else {
      PetscCall(MatCreateSeqDenseCUPM<T>(PetscObjectComm(PetscObjectCast(A)), n, m, device_array, &cmat, dctx));
    }
  }
  PetscCall(MatDenseSetLDA(cmat, mimpl->lda));
  // place CPU array if present but do not copy any data
  if (const auto host_array = mimpl->v) {
    cmat->offloadmask = PETSC_OFFLOAD_GPU;
    PetscCall(MatDensePlaceArray(cmat, array_offset(host_array)));
  }

  cmat->offloadmask = A->offloadmask;
  *mat              = cmat;
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <device::cupm::DeviceType T>
inline PetscErrorCode MatDense_Seq_CUPM<T>::RestoreSubMatrix(Mat A, Mat *m) noexcept
{
  const auto mimpl = MatIMPLCast(A);
  const auto cmat  = mimpl->cmat;
  const auto reset = static_cast<bool>(mimpl->v);
  bool       copy, was_offload_host;

  PetscFunctionBegin;
  PetscCheck(mimpl->matinuse, PETSC_COMM_SELF, PETSC_ERR_ORDER, "Need to call MatDenseGetSubMatrix() first");
  PetscCheck(cmat, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Missing internal column matrix");
  PetscCheck(*m == cmat, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Not the matrix obtained from MatDenseGetSubMatrix()");
  mimpl->matinuse = 0;

  // calls to ResetArray may change it, so save it here
  was_offload_host = cmat->offloadmask == PETSC_OFFLOAD_CPU;
  if (was_offload_host && !reset) {
    copy = true;
    PetscCall(MatSeqDenseSetPreallocation(A, nullptr));
  } else {
    copy = false;
  }

  PetscCall(ResetArray(cmat));
  if (reset) PetscCall(MatDenseResetArray(cmat));
  if (copy) {
    PetscDeviceContext dctx;

    PetscCall(GetHandles_(&dctx));
    PetscCall(DeviceToHost_(A, dctx));
  } else {
    A->offloadmask = was_offload_host ? PETSC_OFFLOAD_CPU : PETSC_OFFLOAD_GPU;
  }

  cmat->offloadmask = PETSC_OFFLOAD_UNALLOCATED;
  *m                = nullptr;
  PetscFunctionReturn(PETSC_SUCCESS);
}

// ==========================================================================================

namespace
{

template <device::cupm::DeviceType T>
inline PetscErrorCode MatMatMultNumeric_SeqDenseCUPM_SeqDenseCUPM(Mat A, Mat B, Mat C, PetscBool TA, PetscBool TB) noexcept
{
  PetscFunctionBegin;
  if (TA) {
    if (TB) {
      PetscCall(MatDense_Seq_CUPM<T>::template MatMatMult_Numeric_Dispatch<true, true>(A, B, C));
    } else {
      PetscCall(MatDense_Seq_CUPM<T>::template MatMatMult_Numeric_Dispatch<true, false>(A, B, C));
    }
  } else {
    if (TB) {
      PetscCall(MatDense_Seq_CUPM<T>::template MatMatMult_Numeric_Dispatch<false, true>(A, B, C));
    } else {
      PetscCall(MatDense_Seq_CUPM<T>::template MatMatMult_Numeric_Dispatch<false, false>(A, B, C));
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <device::cupm::DeviceType T>
inline PetscErrorCode MatSolverTypeRegister_DENSECUPM() noexcept
{
  PetscFunctionBegin;
  for (auto ftype : util::make_array(MAT_FACTOR_LU, MAT_FACTOR_CHOLESKY, MAT_FACTOR_QR)) {
    PetscCall(MatSolverTypeRegister(MatDense_Seq_CUPM<T>::MATSOLVERCUPM(), MATSEQDENSE, ftype, MatDense_Seq_CUPM<T>::GetFactor));
    PetscCall(MatSolverTypeRegister(MatDense_Seq_CUPM<T>::MATSOLVERCUPM(), MatDense_Seq_CUPM<T>::MATSEQDENSECUPM(), ftype, MatDense_Seq_CUPM<T>::GetFactor));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

} // anonymous namespace

} // namespace impl

} // namespace cupm

} // namespace mat

} // namespace Petsc

#endif // __cplusplus

#endif // PETSCMATSEQDENSECUPM_HPP

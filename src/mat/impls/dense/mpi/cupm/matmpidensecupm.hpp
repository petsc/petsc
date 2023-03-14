#ifndef PETSCMATMPIDENSECUPM_HPP
#define PETSCMATMPIDENSECUPM_HPP

#include <petsc/private/matdensecupmimpl.h> /*I <petscmat.h> I*/
#include <../src/mat/impls/dense/mpi/mpidense.h>

#ifdef __cplusplus
  #include <../src/mat/impls/dense/seq/cupm/matseqdensecupm.hpp>
  #include <../src/vec/vec/impls/mpi/cupm/vecmpicupm.hpp>

namespace Petsc
{

namespace mat
{

namespace cupm
{

namespace impl
{

template <device::cupm::DeviceType T>
class MatDense_MPI_CUPM : MatDense_CUPM<T, MatDense_MPI_CUPM<T>> {
public:
  MATDENSECUPM_HEADER(T, MatDense_MPI_CUPM<T>);

private:
  PETSC_NODISCARD static constexpr Mat_MPIDense *MatIMPLCast_(Mat) noexcept;
  PETSC_NODISCARD static constexpr MatType       MATIMPLCUPM_() noexcept;

  static PetscErrorCode SetPreallocation_(Mat, PetscDeviceContext, PetscScalar *) noexcept;

  template <bool to_host>
  static PetscErrorCode Convert_Dispatch_(Mat, MatType, MatReuse, Mat *) noexcept;

public:
  PETSC_NODISCARD static constexpr const char *MatConvert_mpidensecupm_mpidense_C() noexcept;

  PETSC_NODISCARD static constexpr const char *MatProductSetFromOptions_mpiaij_mpidensecupm_C() noexcept;
  PETSC_NODISCARD static constexpr const char *MatProductSetFromOptions_mpidensecupm_mpiaij_C() noexcept;

  PETSC_NODISCARD static constexpr const char *MatProductSetFromOptions_mpiaijcupmsparse_mpidensecupm_C() noexcept;
  PETSC_NODISCARD static constexpr const char *MatProductSetFromOptions_mpidensecupm_mpiaijcupmsparse_C() noexcept;

  static PetscErrorCode Create(Mat) noexcept;

  static PetscErrorCode BindToCPU(Mat, PetscBool) noexcept;
  static PetscErrorCode Convert_MPIDenseCUPM_MPIDense(Mat, MatType, MatReuse, Mat *) noexcept;
  static PetscErrorCode Convert_MPIDense_MPIDenseCUPM(Mat, MatType, MatReuse, Mat *) noexcept;

  template <PetscMemType, PetscMemoryAccessMode>
  static PetscErrorCode GetArray(Mat, PetscScalar **, PetscDeviceContext = nullptr) noexcept;
  template <PetscMemType, PetscMemoryAccessMode>
  static PetscErrorCode RestoreArray(Mat, PetscScalar **, PetscDeviceContext = nullptr) noexcept;

private:
  template <PetscMemType mtype, PetscMemoryAccessMode mode>
  static PetscErrorCode GetArrayC_(Mat m, PetscScalar **p) noexcept
  {
    return GetArray<mtype, mode>(m, p);
  }

  template <PetscMemType mtype, PetscMemoryAccessMode mode>
  static PetscErrorCode RestoreArrayC_(Mat m, PetscScalar **p) noexcept
  {
    return RestoreArray<mtype, mode>(m, p);
  }

public:
  template <PetscMemoryAccessMode>
  static PetscErrorCode GetColumnVec(Mat, PetscInt, Vec *) noexcept;
  template <PetscMemoryAccessMode>
  static PetscErrorCode RestoreColumnVec(Mat, PetscInt, Vec *) noexcept;

  static PetscErrorCode PlaceArray(Mat, const PetscScalar *) noexcept;
  static PetscErrorCode ReplaceArray(Mat, const PetscScalar *) noexcept;
  static PetscErrorCode ResetArray(Mat) noexcept;

  static PetscErrorCode Shift(Mat, PetscScalar) noexcept;
};

} // namespace impl

namespace
{

// Declare this here so that the functions below can make use of it
template <device::cupm::DeviceType T>
inline PetscErrorCode MatCreateMPIDenseCUPM(MPI_Comm comm, PetscInt m, PetscInt n, PetscInt M, PetscInt N, PetscScalar *data, Mat *A, PetscDeviceContext dctx = nullptr, bool preallocate = true) noexcept
{
  PetscFunctionBegin;
  PetscCall(impl::MatDense_MPI_CUPM<T>::CreateIMPLDenseCUPM(comm, m, n, M, N, data, A, dctx, preallocate));
  PetscFunctionReturn(PETSC_SUCCESS);
}

} // anonymous namespace

namespace impl
{

// ==========================================================================================
// MatDense_MPI_CUPM -- Private API
// ==========================================================================================

template <device::cupm::DeviceType T>
inline constexpr Mat_MPIDense *MatDense_MPI_CUPM<T>::MatIMPLCast_(Mat m) noexcept
{
  return static_cast<Mat_MPIDense *>(m->data);
}

template <device::cupm::DeviceType T>
inline constexpr MatType MatDense_MPI_CUPM<T>::MATIMPLCUPM_() noexcept
{
  return MATMPIDENSECUPM();
}

// ==========================================================================================

template <device::cupm::DeviceType T>
inline PetscErrorCode MatDense_MPI_CUPM<T>::SetPreallocation_(Mat A, PetscDeviceContext dctx, PetscScalar *device_array) noexcept
{
  PetscFunctionBegin;
  if (auto &mimplA = MatIMPLCast(A)->A) {
    PetscCall(MatSetType(mimplA, MATSEQDENSECUPM()));
    PetscCall(MatDense_Seq_CUPM<T>::SetPreallocation(mimplA, dctx, device_array));
  } else {
    PetscCall(MatCreateSeqDenseCUPM<T>(PETSC_COMM_SELF, A->rmap->n, A->cmap->N, device_array, &mimplA, dctx));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <device::cupm::DeviceType T>
template <bool to_host>
inline PetscErrorCode MatDense_MPI_CUPM<T>::Convert_Dispatch_(Mat M, MatType, MatReuse reuse, Mat *newmat) noexcept
{
  PetscFunctionBegin;
  if (reuse == MAT_INITIAL_MATRIX) {
    PetscCall(MatDuplicate(M, MAT_COPY_VALUES, newmat));
  } else if (reuse == MAT_REUSE_MATRIX) {
    PetscCall(MatCopy(M, *newmat, SAME_NONZERO_PATTERN));
  }
  {
    const auto B    = *newmat;
    const auto pobj = PetscObjectCast(B);

    if (to_host) {
      PetscCall(BindToCPU(B, PETSC_TRUE));
    } else {
      PetscCall(PetscDeviceInitialize(PETSC_DEVICE_CUPM()));
    }

    PetscCall(PetscStrFreeAllocpy(to_host ? VECSTANDARD : VecMPI_CUPM::VECCUPM(), &B->defaultvectype));
    PetscCall(PetscObjectChangeTypeName(pobj, to_host ? MATMPIDENSE : MATMPIDENSECUPM()));

    // ============================================================
    // Composed Ops
    // ============================================================
    MatComposeOp_CUPM(to_host, pobj, MatConvert_mpidensecupm_mpidense_C(), nullptr, Convert_MPIDenseCUPM_MPIDense);
    MatComposeOp_CUPM(to_host, pobj, MatProductSetFromOptions_mpiaij_mpidensecupm_C(), nullptr, MatProductSetFromOptions_MPIAIJ_MPIDense);
    MatComposeOp_CUPM(to_host, pobj, MatProductSetFromOptions_mpiaijcupmsparse_mpidensecupm_C(), nullptr, MatProductSetFromOptions_MPIAIJ_MPIDense);
    MatComposeOp_CUPM(to_host, pobj, MatProductSetFromOptions_mpidensecupm_mpiaij_C(), nullptr, MatProductSetFromOptions_MPIDense_MPIAIJ);
    MatComposeOp_CUPM(to_host, pobj, MatProductSetFromOptions_mpidensecupm_mpiaijcupmsparse_C(), nullptr, MatProductSetFromOptions_MPIDense_MPIAIJ);
    MatComposeOp_CUPM(to_host, pobj, MatDenseCUPMGetArray_C(), nullptr, GetArrayC_<PETSC_MEMTYPE_DEVICE, PETSC_MEMORY_ACCESS_READ_WRITE>);
    MatComposeOp_CUPM(to_host, pobj, MatDenseCUPMGetArrayRead_C(), nullptr, GetArrayC_<PETSC_MEMTYPE_DEVICE, PETSC_MEMORY_ACCESS_READ>);
    MatComposeOp_CUPM(to_host, pobj, MatDenseCUPMGetArrayWrite_C(), nullptr, GetArrayC_<PETSC_MEMTYPE_DEVICE, PETSC_MEMORY_ACCESS_WRITE>);
    MatComposeOp_CUPM(to_host, pobj, MatDenseCUPMRestoreArray_C(), nullptr, RestoreArrayC_<PETSC_MEMTYPE_DEVICE, PETSC_MEMORY_ACCESS_READ_WRITE>);
    MatComposeOp_CUPM(to_host, pobj, MatDenseCUPMRestoreArrayRead_C(), nullptr, RestoreArrayC_<PETSC_MEMTYPE_DEVICE, PETSC_MEMORY_ACCESS_READ>);
    MatComposeOp_CUPM(to_host, pobj, MatDenseCUPMRestoreArrayWrite_C(), nullptr, RestoreArrayC_<PETSC_MEMTYPE_DEVICE, PETSC_MEMORY_ACCESS_WRITE>);
    MatComposeOp_CUPM(to_host, pobj, MatDenseCUPMPlaceArray_C(), nullptr, PlaceArray);
    MatComposeOp_CUPM(to_host, pobj, MatDenseCUPMResetArray_C(), nullptr, ResetArray);
    MatComposeOp_CUPM(to_host, pobj, MatDenseCUPMReplaceArray_C(), nullptr, ReplaceArray);

    if (to_host) {
      if (auto &m_A = MatIMPLCast(B)->A) PetscCall(MatConvert(m_A, MATSEQDENSE, MAT_INPLACE_MATRIX, &m_A));
      B->offloadmask = PETSC_OFFLOAD_CPU;
    } else {
      if (auto &m_A = MatIMPLCast(B)->A) {
        PetscCall(MatConvert(m_A, MATSEQDENSECUPM(), MAT_INPLACE_MATRIX, &m_A));
        B->offloadmask = PETSC_OFFLOAD_BOTH;
      } else {
        B->offloadmask = PETSC_OFFLOAD_UNALLOCATED;
      }
      PetscCall(BindToCPU(B, PETSC_FALSE));
    }

    // ============================================================
    // Function Pointer Ops
    // ============================================================
    MatSetOp_CUPM(to_host, B, bindtocpu, nullptr, BindToCPU);
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

// ==========================================================================================
// MatDense_MPI_CUPM -- Public API
// ==========================================================================================

template <device::cupm::DeviceType T>
inline constexpr const char *MatDense_MPI_CUPM<T>::MatConvert_mpidensecupm_mpidense_C() noexcept
{
  return T == device::cupm::DeviceType::CUDA ? "MatConvert_mpidensecuda_mpidense_C" : "MatConvert_mpidensehip_mpidense_C";
}

template <device::cupm::DeviceType T>
inline constexpr const char *MatDense_MPI_CUPM<T>::MatProductSetFromOptions_mpiaij_mpidensecupm_C() noexcept
{
  return T == device::cupm::DeviceType::CUDA ? "MatProductSetFromOptions_mpiaij_mpidensecuda_C" : "MatProductSetFromOptions_mpiaij_mpidensehip_C";
}

template <device::cupm::DeviceType T>
inline constexpr const char *MatDense_MPI_CUPM<T>::MatProductSetFromOptions_mpidensecupm_mpiaij_C() noexcept
{
  return T == device::cupm::DeviceType::CUDA ? "MatProductSetFromOptions_mpidensecuda_mpiaij_C" : "MatProductSetFromOptions_mpidensehip_mpiaij_C";
}

template <device::cupm::DeviceType T>
inline constexpr const char *MatDense_MPI_CUPM<T>::MatProductSetFromOptions_mpiaijcupmsparse_mpidensecupm_C() noexcept
{
  return T == device::cupm::DeviceType::CUDA ? "MatProductSetFromOptions_mpiaijcusparse_mpidensecuda_C" : "MatProductSetFromOptions_mpiaijhipsparse_mpidensehip_C";
}

template <device::cupm::DeviceType T>
inline constexpr const char *MatDense_MPI_CUPM<T>::MatProductSetFromOptions_mpidensecupm_mpiaijcupmsparse_C() noexcept
{
  return T == device::cupm::DeviceType::CUDA ? "MatProductSetFromOptions_mpidensecuda_mpiaijcusparse_C" : "MatProductSetFromOptions_mpidensehip_mpiaijhipsparse_C";
}

// ==========================================================================================

template <device::cupm::DeviceType T>
inline PetscErrorCode MatDense_MPI_CUPM<T>::Create(Mat A) noexcept
{
  PetscFunctionBegin;
  PetscCall(MatCreate_MPIDense(A));
  PetscCall(Convert_MPIDense_MPIDenseCUPM(A, MATMPIDENSECUPM(), MAT_INPLACE_MATRIX, &A));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// ==========================================================================================

template <device::cupm::DeviceType T>
inline PetscErrorCode MatDense_MPI_CUPM<T>::BindToCPU(Mat A, PetscBool usehost) noexcept
{
  const auto mimpl = MatIMPLCast(A);
  const auto pobj  = PetscObjectCast(A);

  PetscFunctionBegin;
  PetscCheck(!mimpl->vecinuse, PetscObjectComm(pobj), PETSC_ERR_ORDER, "Need to call MatDenseRestoreColumnVec() first");
  PetscCheck(!mimpl->matinuse, PetscObjectComm(pobj), PETSC_ERR_ORDER, "Need to call MatDenseRestoreSubMatrix() first");
  if (const auto mimpl_A = mimpl->A) PetscCall(MatBindToCPU(mimpl_A, usehost));
  A->boundtocpu = usehost;
  PetscCall(PetscStrFreeAllocpy(usehost ? PETSCRANDER48 : PETSCDEVICERAND(), &A->defaultrandtype));
  if (!usehost) {
    PetscBool iscupm;

    PetscCall(PetscObjectTypeCompare(PetscObjectCast(mimpl->cvec), VecMPI_CUPM::VECMPICUPM(), &iscupm));
    if (!iscupm) PetscCall(VecDestroy(&mimpl->cvec));
    PetscCall(PetscObjectTypeCompare(PetscObjectCast(mimpl->cmat), MATMPIDENSECUPM(), &iscupm));
    if (!iscupm) PetscCall(MatDestroy(&mimpl->cmat));
  }

  MatComposeOp_CUPM(usehost, pobj, "MatDenseGetColumnVec_C", MatDenseGetColumnVec_MPIDense, GetColumnVec<PETSC_MEMORY_ACCESS_READ_WRITE>);
  MatComposeOp_CUPM(usehost, pobj, "MatDenseRestoreColumnVec_C", MatDenseRestoreColumnVec_MPIDense, RestoreColumnVec<PETSC_MEMORY_ACCESS_READ_WRITE>);
  MatComposeOp_CUPM(usehost, pobj, "MatDenseGetColumnVecRead_C", MatDenseGetColumnVecRead_MPIDense, GetColumnVec<PETSC_MEMORY_ACCESS_READ>);
  MatComposeOp_CUPM(usehost, pobj, "MatDenseRestoreColumnVecRead_C", MatDenseRestoreColumnVecRead_MPIDense, RestoreColumnVec<PETSC_MEMORY_ACCESS_READ>);
  MatComposeOp_CUPM(usehost, pobj, "MatDenseGetColumnVecWrite_C", MatDenseGetColumnVecWrite_MPIDense, GetColumnVec<PETSC_MEMORY_ACCESS_WRITE>);
  MatComposeOp_CUPM(usehost, pobj, "MatDenseRestoreColumnVecWrite_C", MatDenseRestoreColumnVecWrite_MPIDense, RestoreColumnVec<PETSC_MEMORY_ACCESS_WRITE>);

  MatSetOp_CUPM(usehost, A, shift, MatShift_MPIDense, Shift);

  if (const auto mimpl_cmat = mimpl->cmat) PetscCall(MatBindToCPU(mimpl_cmat, usehost));
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <device::cupm::DeviceType T>
inline PetscErrorCode MatDense_MPI_CUPM<T>::Convert_MPIDenseCUPM_MPIDense(Mat M, MatType mtype, MatReuse reuse, Mat *newmat) noexcept
{
  PetscFunctionBegin;
  PetscCall(Convert_Dispatch_</* to host */ true>(M, mtype, reuse, newmat));
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <device::cupm::DeviceType T>
inline PetscErrorCode MatDense_MPI_CUPM<T>::Convert_MPIDense_MPIDenseCUPM(Mat M, MatType mtype, MatReuse reuse, Mat *newmat) noexcept
{
  PetscFunctionBegin;
  PetscCall(Convert_Dispatch_</* to host */ false>(M, mtype, reuse, newmat));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// ==========================================================================================

template <device::cupm::DeviceType T>
template <PetscMemType, PetscMemoryAccessMode access>
inline PetscErrorCode MatDense_MPI_CUPM<T>::GetArray(Mat A, PetscScalar **array, PetscDeviceContext) noexcept
{
  PetscFunctionBegin;
  PetscCall(MatDenseCUPMGetArray_Private<T, access>(MatIMPLCast(A)->A, array));
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <device::cupm::DeviceType T>
template <PetscMemType, PetscMemoryAccessMode access>
inline PetscErrorCode MatDense_MPI_CUPM<T>::RestoreArray(Mat A, PetscScalar **array, PetscDeviceContext) noexcept
{
  PetscFunctionBegin;
  PetscCall(MatDenseCUPMRestoreArray_Private<T, access>(MatIMPLCast(A)->A, array));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// ==========================================================================================

template <device::cupm::DeviceType T>
template <PetscMemoryAccessMode access>
inline PetscErrorCode MatDense_MPI_CUPM<T>::GetColumnVec(Mat A, PetscInt col, Vec *v) noexcept
{
  using namespace vec::cupm;

  const auto mimpl   = MatIMPLCast(A);
  const auto mimpl_A = mimpl->A;
  const auto pobj    = PetscObjectCast(A);
  auto      &cvec    = mimpl->cvec;
  PetscInt   lda;

  PetscFunctionBegin;
  PetscCheck(!mimpl->vecinuse, PetscObjectComm(pobj), PETSC_ERR_ORDER, "Need to call MatDenseRestoreColumnVec() first");
  PetscCheck(!mimpl->matinuse, PetscObjectComm(pobj), PETSC_ERR_ORDER, "Need to call MatDenseRestoreSubMatrix() first");
  mimpl->vecinuse = col + 1;

  if (!cvec) PetscCall(VecCreateMPICUPMWithArray<T>(PetscObjectComm(pobj), A->rmap->bs, A->rmap->n, A->rmap->N, nullptr, &cvec));

  PetscCall(MatDenseGetLDA(mimpl_A, &lda));
  PetscCall(MatDenseCUPMGetArray_Private<T, access>(mimpl_A, const_cast<PetscScalar **>(&mimpl->ptrinuse)));
  PetscCall(VecCUPMPlaceArrayAsync<T>(cvec, mimpl->ptrinuse + static_cast<std::size_t>(col) * static_cast<std::size_t>(lda)));

  if (access == PETSC_MEMORY_ACCESS_READ) PetscCall(VecLockReadPush(cvec));
  *v = cvec;
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <device::cupm::DeviceType T>
template <PetscMemoryAccessMode access>
inline PetscErrorCode MatDense_MPI_CUPM<T>::RestoreColumnVec(Mat A, PetscInt, Vec *v) noexcept
{
  using namespace vec::cupm;

  const auto mimpl = MatIMPLCast(A);
  const auto cvec  = mimpl->cvec;

  PetscFunctionBegin;
  PetscCheck(mimpl->vecinuse, PETSC_COMM_SELF, PETSC_ERR_ORDER, "Need to call MatDenseGetColumnVec() first");
  PetscCheck(cvec, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Missing internal column vector");
  mimpl->vecinuse = 0;

  PetscCall(MatDenseCUPMRestoreArray_Private<T, access>(mimpl->A, const_cast<PetscScalar **>(&mimpl->ptrinuse)));
  if (access == PETSC_MEMORY_ACCESS_READ) PetscCall(VecLockReadPop(cvec));
  PetscCall(VecCUPMResetArrayAsync<T>(cvec));

  if (v) *v = nullptr;
  PetscFunctionReturn(PETSC_SUCCESS);
}

// ==========================================================================================

template <device::cupm::DeviceType T>
inline PetscErrorCode MatDense_MPI_CUPM<T>::PlaceArray(Mat A, const PetscScalar *array) noexcept
{
  const auto mimpl = MatIMPLCast(A);

  PetscFunctionBegin;
  PetscCheck(!mimpl->vecinuse, PetscObjectComm(PetscObjectCast(A)), PETSC_ERR_ORDER, "Need to call MatDenseRestoreColumnVec() first");
  PetscCheck(!mimpl->matinuse, PetscObjectComm(PetscObjectCast(A)), PETSC_ERR_ORDER, "Need to call MatDenseRestoreSubMatrix() first");
  PetscCall(MatDenseCUPMPlaceArray<T>(mimpl->A, array));
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <device::cupm::DeviceType T>
inline PetscErrorCode MatDense_MPI_CUPM<T>::ReplaceArray(Mat A, const PetscScalar *array) noexcept
{
  const auto mimpl = MatIMPLCast(A);

  PetscFunctionBegin;
  PetscCheck(!mimpl->vecinuse, PetscObjectComm(PetscObjectCast(A)), PETSC_ERR_ORDER, "Need to call MatDenseRestoreColumnVec() first");
  PetscCheck(!mimpl->matinuse, PetscObjectComm(PetscObjectCast(A)), PETSC_ERR_ORDER, "Need to call MatDenseRestoreSubMatrix() first");
  PetscCall(MatDenseCUPMReplaceArray<T>(mimpl->A, array));
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <device::cupm::DeviceType T>
inline PetscErrorCode MatDense_MPI_CUPM<T>::ResetArray(Mat A) noexcept
{
  const auto mimpl = MatIMPLCast(A);

  PetscFunctionBegin;
  PetscCheck(!mimpl->vecinuse, PetscObjectComm(PetscObjectCast(A)), PETSC_ERR_ORDER, "Need to call MatDenseRestoreColumnVec() first");
  PetscCheck(!mimpl->matinuse, PetscObjectComm(PetscObjectCast(A)), PETSC_ERR_ORDER, "Need to call MatDenseRestoreSubMatrix() first");
  PetscCall(MatDenseCUPMResetArray<T>(mimpl->A));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// ==========================================================================================

template <device::cupm::DeviceType T>
inline PetscErrorCode MatDense_MPI_CUPM<T>::Shift(Mat A, PetscScalar alpha) noexcept
{
  PetscDeviceContext dctx;

  PetscFunctionBegin;
  PetscCall(GetHandles_(&dctx));
  PetscCall(PetscInfo(A, "Performing Shift on backend\n"));
  PetscCall(DiagonalUnaryTransform(A, A->rmap->rstart, A->rmap->rend, A->cmap->N, dctx, device::cupm::functors::make_plus_equals(alpha)));
  PetscFunctionReturn(PETSC_SUCCESS);
}

} // namespace impl

namespace
{

template <device::cupm::DeviceType T>
inline PetscErrorCode MatCreateDenseCUPM(MPI_Comm comm, PetscInt n, PetscInt m, PetscInt N, PetscInt M, PetscScalar *data, Mat *A, PetscDeviceContext dctx = nullptr) noexcept
{
  PetscMPIInt size;

  PetscFunctionBegin;
  PetscValidPointer(A, 7);
  PetscCallMPI(MPI_Comm_size(comm, &size));
  if (size > 1) {
    PetscCall(MatCreateMPIDenseCUPM<T>(comm, n, m, N, M, data, A, dctx));
  } else {
    if (n == PETSC_DECIDE) n = N;
    if (m == PETSC_DECIDE) m = M;
    // It's OK here if both are PETSC_DECIDE since PetscSplitOwnership() will catch that down
    // the line
    PetscCall(MatCreateSeqDenseCUPM<T>(comm, n, m, data, A, dctx));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

} // anonymous namespace

} // namespace cupm

} // namespace mat

} // namespace Petsc

#endif // __cplusplus

#endif // PETSCMATMPIDENSECUPM_HPP

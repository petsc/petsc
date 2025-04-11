#include <../src/ksp/ksp/utils/lmvm/dense/denseqn.h> /*I "petscksp.h" I*/
#include <petscblaslapack.h>
#include <petscmat.h>
#include <petscsys.h>
#include <petscsystypes.h>
#include <petscis.h>
#include <petscoptions.h>
#include <petscdevice.h>
#include <petsc/private/deviceimpl.h>

const char *const MatLMVMDenseTypes[] = {"reorder", "inplace", "MatLMVMDenseType", "MAT_LMVM_DENSE_", NULL};

PETSC_INTERN PetscErrorCode VecCyclicShift(Mat B, Vec X, PetscInt d, Vec cyclic_work_vec)
{
  Mat_LMVM          *lmvm = (Mat_LMVM *)B->data;
  PetscInt           m    = lmvm->m;
  PetscInt           n;
  const PetscScalar *src;
  PetscScalar       *dest;
  PetscMemType       src_memtype;
  PetscMemType       dest_memtype;

  PetscFunctionBegin;
  PetscCall(VecGetLocalSize(X, &n));
  if (!cyclic_work_vec) PetscCall(VecDuplicate(X, &cyclic_work_vec));
  PetscCall(VecCopy(X, cyclic_work_vec));
  PetscCall(VecGetArrayReadAndMemType(cyclic_work_vec, &src, &src_memtype));
  PetscCall(VecGetArrayWriteAndMemType(X, &dest, &dest_memtype));
  if (n == 0) { /* no work on this process */
    PetscCall(VecRestoreArrayWriteAndMemType(X, &dest));
    PetscCall(VecRestoreArrayReadAndMemType(cyclic_work_vec, &src));
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  PetscAssert(src_memtype == dest_memtype, PETSC_COMM_SELF, PETSC_ERR_PLIB, "memtype of duplicate does not match");
  if (PetscMemTypeHost(src_memtype)) {
    PetscCall(PetscArraycpy(dest, &src[d], m - d));
    PetscCall(PetscArraycpy(&dest[m - d], src, d));
  } else {
    PetscDeviceContext dctx;

    PetscCall(PetscDeviceContextGetCurrentContext(&dctx));
    PetscCall(PetscDeviceRegisterMemory(dest, dest_memtype, m * sizeof(*dest)));
    PetscCall(PetscDeviceRegisterMemory(src, src_memtype, m * sizeof(*src)));
    PetscCall(PetscDeviceArrayCopy(dctx, dest, &src[d], m - d));
    PetscCall(PetscDeviceArrayCopy(dctx, &dest[m - d], src, d));
  }
  PetscCall(VecRestoreArrayWriteAndMemType(X, &dest));
  PetscCall(VecRestoreArrayReadAndMemType(cyclic_work_vec, &src));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static inline PetscInt recycle_index(PetscInt m, PetscInt idx)
{
  return idx % m;
}

static inline PetscInt oldest_update(PetscInt m, PetscInt idx)
{
  return PetscMax(0, idx - m);
}

PETSC_INTERN PetscErrorCode VecRecycleOrderToHistoryOrder(Mat B, Vec X, PetscInt num_updates, Vec cyclic_work_vec)
{
  Mat_LMVM *lmvm = (Mat_LMVM *)B->data;
  PetscInt  m    = lmvm->m;
  PetscInt  oldest_index;

  PetscFunctionBegin;
  oldest_index = recycle_index(m, oldest_update(m, num_updates));
  if (oldest_index == 0) PetscFunctionReturn(PETSC_SUCCESS); /* vector is already in history order */
  PetscCall(VecCyclicShift(B, X, oldest_index, cyclic_work_vec));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode VecHistoryOrderToRecycleOrder(Mat B, Vec X, PetscInt num_updates, Vec cyclic_work_vec)
{
  Mat_LMVM *lmvm = (Mat_LMVM *)B->data;
  PetscInt  m    = lmvm->m;
  PetscInt  oldest_index;

  PetscFunctionBegin;
  oldest_index = recycle_index(m, oldest_update(m, num_updates));
  if (oldest_index == 0) PetscFunctionReturn(PETSC_SUCCESS); /* vector is already in recycle order */
  PetscCall(VecCyclicShift(B, X, m - oldest_index, cyclic_work_vec));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode MatUpperTriangularSolveInPlace_Internal(MatLMVMDenseType type, PetscMemType memtype, PetscBool hermitian_transpose, PetscInt m, PetscInt oldest, PetscInt next, const PetscScalar A[], PetscInt lda, PetscScalar x[], PetscInt stride)
{
  PetscInt oldest_index = oldest % m;
  PetscInt next_index   = (next - 1) % m + 1;
  PetscInt N            = next - oldest;

  PetscFunctionBegin;
  /* if oldest_index == 0, the two strategies are equivalent, redirect to the simpler one */
  if (oldest_index == 0) type = MAT_LMVM_DENSE_REORDER;
  switch (type) {
  case MAT_LMVM_DENSE_REORDER:
    if (PetscMemTypeHost(memtype)) {
      PetscBLASInt n, lda_blas, one = 1;
      PetscCall(PetscBLASIntCast(N, &n));
      PetscCall(PetscBLASIntCast(lda, &lda_blas));
      PetscCallBLAS("BLAStrsv", BLAStrsv_("U", hermitian_transpose ? "C" : "N", "NotUnitTriangular", &n, A, &lda_blas, x, &one));
      PetscCall(PetscLogFlops(1.0 * n * n));
#if defined(PETSC_HAVE_CUPM)
    } else if (PetscMemTypeDevice(memtype)) {
      PetscCall(MatUpperTriangularSolveInPlace_CUPM(hermitian_transpose, N, A, lda, x, 1));
#endif
    } else SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Unsupported memtype");
    break;
  case MAT_LMVM_DENSE_INPLACE:
    if (PetscMemTypeHost(memtype)) {
      PetscBLASInt n_old, n_new, lda_blas, one = 1;
      PetscScalar  minus_one = -1.0;
      PetscScalar  sone      = 1.0;
      PetscCall(PetscBLASIntCast(m - oldest_index, &n_old));
      PetscCall(PetscBLASIntCast(next_index, &n_new));
      PetscCall(PetscBLASIntCast(lda, &lda_blas));
      if (!hermitian_transpose) {
        if (n_new > 0) PetscCallBLAS("BLAStrsv", BLAStrsv_("U", "N", "NotUnitTriangular", &n_new, A, &lda_blas, x, &one));
        if (n_new > 0 && n_old > 0) PetscCallBLAS("BLASgemv", BLASgemv_("N", &n_old, &n_new, &minus_one, &A[oldest_index], &lda_blas, x, &one, &sone, &x[oldest_index], &one));
        if (n_old > 0) PetscCallBLAS("BLAStrsv", BLAStrsv_("U", "N", "NotUnitTriangular", &n_old, &A[oldest_index * (lda + 1)], &lda_blas, &x[oldest_index], &one));
      } else {
        if (n_old > 0) {
          PetscCallBLAS("BLAStrsv", BLAStrsv_("U", "C", "NotUnitTriangular", &n_old, &A[oldest_index * (lda + 1)], &lda_blas, &x[oldest_index], &one));
          if (n_new > 0 && n_old > 0) PetscCallBLAS("BLASgemv", BLASgemv_("C", &n_old, &n_new, &minus_one, &A[oldest_index], &lda_blas, &x[oldest_index], &one, &sone, x, &one));
        }
        if (n_new > 0) PetscCallBLAS("BLAStrsv", BLAStrsv_("U", "C", "NotUnitTriangular", &n_new, A, &lda_blas, x, &one));
      }
      PetscCall(PetscLogFlops(1.0 * N * N));
#if defined(PETSC_HAVE_CUPM)
    } else if (PetscMemTypeDevice(memtype)) {
      PetscCall(MatUpperTriangularSolveInPlaceCyclic_CUPM(hermitian_transpose, m, oldest, next, A, lda, x, stride));
#endif
    } else SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Unsupported memtype");
    break;
  default:
    PetscUnreachable();
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode MatUpperTriangularSolveInPlace(Mat B, Mat Amat, Vec X, PetscBool hermitian_transpose, PetscInt num_updates, MatLMVMDenseType strategy)
{
  Mat_LMVM          *lmvm = (Mat_LMVM *)B->data;
  PetscInt           m    = lmvm->m;
  PetscInt           h, local_n;
  PetscInt           lda;
  PetscScalar       *x;
  PetscMemType       memtype_r, memtype_x;
  const PetscScalar *A;

  PetscFunctionBegin;
  h = num_updates - oldest_update(m, num_updates);
  if (!h) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(VecGetLocalSize(X, &local_n));
  PetscCall(VecGetArrayAndMemType(X, &x, &memtype_x));
  PetscCall(MatDenseGetArrayReadAndMemType(Amat, &A, &memtype_r));
  if (!local_n) {
    PetscCall(MatDenseRestoreArrayReadAndMemType(Amat, &A));
    PetscCall(VecRestoreArrayAndMemType(X, &x));
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  PetscAssert(memtype_x == memtype_r, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Incompatible device pointers");
  PetscCall(MatDenseGetLDA(Amat, &lda));
  PetscCall(MatUpperTriangularSolveInPlace_Internal(strategy, memtype_x, hermitian_transpose, m, oldest_update(m, num_updates), num_updates, A, lda, x, 1));
  PetscCall(VecRestoreArrayWriteAndMemType(X, &x));
  PetscCall(MatDenseRestoreArrayReadAndMemType(Amat, &A));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Shifts R[end-m_keep:end,end-m_keep:end] to R[0:m_keep, 0:m_keep] */

PETSC_INTERN PetscErrorCode MatMove_LR3(Mat B, Mat R, PetscInt m_keep)
{
  Mat_LMVM *lmvm = (Mat_LMVM *)B->data;
  Mat_DQN  *lqn  = (Mat_DQN *)lmvm->ctx;
  PetscInt  M;
  Mat       mat_local, local_sub, local_temp, temp_sub;

  PetscFunctionBegin;
  if (!lqn->temp_mat) PetscCall(MatDuplicate(R, MAT_SHARE_NONZERO_PATTERN, &lqn->temp_mat));
  PetscCall(MatGetLocalSize(R, &M, NULL));
  if (M == 0) PetscFunctionReturn(PETSC_SUCCESS);

  PetscCall(MatDenseGetLocalMatrix(R, &mat_local));
  PetscCall(MatDenseGetLocalMatrix(lqn->temp_mat, &local_temp));
  PetscCall(MatDenseGetSubMatrix(mat_local, lmvm->m - m_keep, lmvm->m, lmvm->m - m_keep, lmvm->m, &local_sub));
  PetscCall(MatDenseGetSubMatrix(local_temp, lmvm->m - m_keep, lmvm->m, lmvm->m - m_keep, lmvm->m, &temp_sub));
  PetscCall(MatCopy(local_sub, temp_sub, SAME_NONZERO_PATTERN));
  PetscCall(MatDenseRestoreSubMatrix(mat_local, &local_sub));
  PetscCall(MatDenseGetSubMatrix(mat_local, 0, m_keep, 0, m_keep, &local_sub));
  PetscCall(MatCopy(temp_sub, local_sub, SAME_NONZERO_PATTERN));
  PetscCall(MatDenseRestoreSubMatrix(mat_local, &local_sub));
  PetscCall(MatDenseRestoreSubMatrix(local_temp, &temp_sub));
  PetscFunctionReturn(PETSC_SUCCESS);
}

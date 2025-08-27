#include <../src/ksp/ksp/utils/lmvm/dense/denseqn.h> /*I "petscksp.h" I*/
#include <../src/ksp/ksp/utils/lmvm/blas_cyclic/blas_cyclic.h>
#include <petscblaslapack.h>
#include <petscmat.h>
#include <petscsys.h>
#include <petscsystypes.h>
#include <petscis.h>
#include <petscoptions.h>
#include <petscdevice.h>
#include <petsc/private/deviceimpl.h>

static PetscErrorCode MatMult_LMVMDQN(Mat, Vec, Vec);
static PetscErrorCode MatMult_LMVMDBFGS(Mat, Vec, Vec);
static PetscErrorCode MatMult_LMVMDDFP(Mat, Vec, Vec);
static PetscErrorCode MatSolve_LMVMDQN(Mat, Vec, Vec);
static PetscErrorCode MatSolve_LMVMDBFGS(Mat, Vec, Vec);
static PetscErrorCode MatSolve_LMVMDDFP(Mat, Vec, Vec);

static inline PetscInt recycle_index(PetscInt m, PetscInt idx)
{
  return idx % m;
}

static inline PetscInt history_index(PetscInt m, PetscInt num_updates, PetscInt idx)
{
  return (idx - num_updates) + PetscMin(m, num_updates);
}

static inline PetscInt oldest_update(PetscInt m, PetscInt idx)
{
  return PetscMax(0, idx - m);
}

static PetscErrorCode MatView_LMVMDQN(Mat B, PetscViewer pv)
{
  Mat_LMVM *lmvm = (Mat_LMVM *)B->data;
  Mat_DQN  *lqn  = (Mat_DQN *)lmvm->ctx;

  PetscBool isascii;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)pv, PETSCVIEWERASCII, &isascii));
  PetscCall(MatView_LMVM(B, pv));
  PetscCall(SymBroydenRescaleView(lqn->rescale, pv));
  if (isascii) PetscCall(PetscViewerASCIIPrintf(pv, "Counts: S x : %" PetscInt_FMT ", S^T x : %" PetscInt_FMT ", Y x : %" PetscInt_FMT ",  Y^T x: %" PetscInt_FMT "\n", lqn->S_count, lqn->St_count, lqn->Y_count, lqn->Yt_count));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatLMVMDQNResetDestructive(Mat B)
{
  Mat_LMVM *lmvm = (Mat_LMVM *)B->data;
  Mat_DQN  *lqn  = (Mat_DQN *)lmvm->ctx;

  PetscFunctionBegin;
  PetscCall(MatDestroy(&lqn->HY));
  PetscCall(MatDestroy(&lqn->BS));
  PetscCall(MatDestroy(&lqn->StY_triu));
  PetscCall(MatDestroy(&lqn->YtS_triu));
  PetscCall(VecDestroy(&lqn->StFprev));
  PetscCall(VecDestroy(&lqn->Fprev_ref));
  lqn->Fprev_state = 0;
  PetscCall(MatDestroy(&lqn->YtS_triu_strict));
  PetscCall(MatDestroy(&lqn->StY_triu_strict));
  PetscCall(MatDestroy(&lqn->StBS));
  PetscCall(MatDestroy(&lqn->YtHY));
  PetscCall(MatDestroy(&lqn->J));
  PetscCall(MatDestroy(&lqn->temp_mat));
  PetscCall(VecDestroy(&lqn->diag_vec));
  PetscCall(VecDestroy(&lqn->diag_vec_recycle_order));
  PetscCall(VecDestroy(&lqn->inv_diag_vec));
  PetscCall(VecDestroy(&lqn->column_work));
  PetscCall(VecDestroy(&lqn->column_work2));
  PetscCall(VecDestroy(&lqn->rwork1));
  PetscCall(VecDestroy(&lqn->rwork2));
  PetscCall(VecDestroy(&lqn->rwork3));
  PetscCall(VecDestroy(&lqn->rwork2_local));
  PetscCall(VecDestroy(&lqn->rwork3_local));
  PetscCall(VecDestroy(&lqn->cyclic_work_vec));
  PetscCall(VecDestroyVecs(lmvm->m, &lqn->PQ));
  PetscCall(PetscFree(lqn->stp));
  PetscCall(PetscFree(lqn->yts));
  PetscCall(PetscFree(lqn->ytq));
  lqn->allocated = PETSC_FALSE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatReset_LMVMDQN_Internal(Mat B, MatLMVMResetMode mode)
{
  Mat_LMVM *lmvm = (Mat_LMVM *)B->data;
  Mat_DQN  *lqn  = (Mat_DQN *)lmvm->ctx;

  PetscFunctionBegin;
  lqn->watchdog         = 0;
  lqn->needPQ           = PETSC_TRUE;
  lqn->num_updates      = 0;
  lqn->num_mult_updates = 0;
  if (MatLMVMResetClearsBases(mode)) PetscCall(MatLMVMDQNResetDestructive(B));
  else {
    if (lqn->BS) PetscCall(MatZeroEntries(lqn->BS));
    if (lqn->HY) PetscCall(MatZeroEntries(lqn->HY));
    if (lqn->StY_triu) { /* Set to identity by default so it is invertible */
      PetscCall(MatZeroEntries(lqn->StY_triu));
      PetscCall(MatShift(lqn->StY_triu, 1.0));
    }
    if (lqn->YtS_triu) {
      PetscCall(MatZeroEntries(lqn->YtS_triu));
      PetscCall(MatShift(lqn->YtS_triu, 1.0));
    }
    if (lqn->YtS_triu_strict) PetscCall(MatZeroEntries(lqn->YtS_triu_strict));
    if (lqn->StY_triu_strict) PetscCall(MatZeroEntries(lqn->StY_triu_strict));
    if (lqn->StBS) {
      PetscCall(MatZeroEntries(lqn->StBS));
      PetscCall(MatShift(lqn->StBS, 1.0));
    }
    if (lqn->YtHY) {
      PetscCall(MatZeroEntries(lqn->YtHY));
      PetscCall(MatShift(lqn->YtHY, 1.0));
    }
    if (lqn->Fprev_ref) PetscCall(VecDestroy(&lqn->Fprev_ref));
    lqn->Fprev_state = 0;
    if (lqn->StFprev) PetscCall(VecZeroEntries(lqn->StFprev));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatReset_LMVMDQN(Mat B, MatLMVMResetMode mode)
{
  Mat_LMVM *lmvm = (Mat_LMVM *)B->data;
  Mat_DQN  *lqn  = (Mat_DQN *)lmvm->ctx;

  PetscFunctionBegin;
  PetscCall(SymBroydenRescaleReset(B, lqn->rescale, mode));
  PetscCall(MatReset_LMVMDQN_Internal(B, mode));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatAllocate_LMVMDQN_Internal(Mat B)
{
  Mat_LMVM *lmvm = (Mat_LMVM *)B->data;
  Mat_DQN  *lqn  = (Mat_DQN *)lmvm->ctx;

  PetscFunctionBegin;
  if (!lqn->allocated) {
    if (lmvm->m > 0) {
      PetscMPIInt rank;
      PetscInt    n, N, m, M;
      PetscBool   is_dbfgs, is_ddfp, is_dqn;
      VecType     vec_type;
      MPI_Comm    comm  = PetscObjectComm((PetscObject)B);
      Mat         Sfull = lmvm->basis[LMBASIS_S]->vecs;

      PetscCall(PetscObjectTypeCompare((PetscObject)B, MATLMVMDBFGS, &is_dbfgs));
      PetscCall(PetscObjectTypeCompare((PetscObject)B, MATLMVMDDFP, &is_ddfp));
      PetscCall(PetscObjectTypeCompare((PetscObject)B, MATLMVMDQN, &is_dqn));

      PetscCallMPI(MPI_Comm_rank(comm, &rank));
      PetscCall(MatGetSize(B, &N, NULL));
      PetscCall(MatGetLocalSize(B, &n, NULL));
      M = lmvm->m;
      m = (rank == 0) ? M : 0;

      /* For DBFGS: Create data needed for MatSolve() eagerly; data needed for MatMult() will be created on demand
       * For DDFP : Create data needed for MatMult() eagerly; data needed for MatSolve() will be created on demand
       * For DQN  : Create all data eagerly */
      PetscCall(VecGetType(lmvm->Xprev, &vec_type));
      if (is_dqn) {
        PetscCall(MatCreateDenseFromVecType(comm, vec_type, m, m, M, M, -1, NULL, &lqn->StY_triu));
        PetscCall(MatCreateDenseFromVecType(comm, vec_type, m, m, M, M, -1, NULL, &lqn->YtS_triu));
        PetscCall(MatCreateVecs(lqn->StY_triu, &lqn->diag_vec, &lqn->rwork1));
        PetscCall(MatCreateVecs(lqn->StY_triu, &lqn->rwork2, &lqn->rwork3));
      } else if (is_ddfp) {
        PetscCall(MatCreateDenseFromVecType(comm, vec_type, m, m, M, M, -1, NULL, &lqn->YtS_triu));
        PetscCall(MatDuplicate(Sfull, MAT_SHARE_NONZERO_PATTERN, &lqn->HY));
        PetscCall(MatCreateVecs(lqn->YtS_triu, &lqn->diag_vec, &lqn->rwork1));
        PetscCall(MatCreateVecs(lqn->YtS_triu, &lqn->rwork2, &lqn->rwork3));
      } else if (is_dbfgs) {
        PetscCall(MatCreateDenseFromVecType(comm, vec_type, m, m, M, M, -1, NULL, &lqn->StY_triu));
        PetscCall(MatDuplicate(Sfull, MAT_SHARE_NONZERO_PATTERN, &lqn->BS));
        PetscCall(MatCreateVecs(lqn->StY_triu, &lqn->diag_vec, &lqn->rwork1));
        PetscCall(MatCreateVecs(lqn->StY_triu, &lqn->rwork2, &lqn->rwork3));
      } else {
        SETERRQ(PetscObjectComm((PetscObject)B), PETSC_ERR_ARG_INCOMP, "MatAllocate_LMVMDQN is only available for dense derived types. (DBFGS, DDFP, DQN");
      }
      /* initialize StY_triu and YtS_triu to identity, if they exist, so it is invertible */
      if (lqn->StY_triu) {
        PetscCall(MatZeroEntries(lqn->StY_triu));
        PetscCall(MatShift(lqn->StY_triu, 1.0));
      }
      if (lqn->YtS_triu) {
        PetscCall(MatZeroEntries(lqn->YtS_triu));
        PetscCall(MatShift(lqn->YtS_triu, 1.0));
      }
      if (lqn->use_recursive && (is_dbfgs || is_ddfp)) {
        PetscCall(VecDuplicateVecs(lmvm->Xprev, lmvm->m, &lqn->PQ));
        PetscCall(VecDuplicate(lmvm->Xprev, &lqn->column_work2));
        PetscCall(PetscMalloc1(lmvm->m, &lqn->yts));
        if (is_dbfgs) {
          PetscCall(PetscMalloc1(lmvm->m, &lqn->stp));
        } else if (is_ddfp) {
          PetscCall(PetscMalloc1(lmvm->m, &lqn->ytq));
        }
      }
      PetscCall(VecDuplicate(lqn->rwork2, &lqn->cyclic_work_vec));
      PetscCall(VecZeroEntries(lqn->rwork1));
      PetscCall(VecZeroEntries(lqn->rwork2));
      PetscCall(VecZeroEntries(lqn->rwork3));
      PetscCall(VecZeroEntries(lqn->diag_vec));
    }
    PetscCall(VecDuplicate(lmvm->Xprev, &lqn->column_work));
    lqn->allocated = PETSC_TRUE;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatSetUp_LMVMDQN(Mat B)
{
  Mat_LMVM *lmvm = (Mat_LMVM *)B->data;
  Mat_DQN  *lqn  = (Mat_DQN *)lmvm->ctx;

  PetscFunctionBegin;
  PetscCall(MatSetUp_LMVM(B));
  PetscCall(SymBroydenRescaleInitializeJ0(B, lqn->rescale));
  PetscCall(MatAllocate_LMVMDQN_Internal(B));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatSetFromOptions_LMVMDQN(Mat B, PetscOptionItems PetscOptionsObject)
{
  Mat_LMVM *lmvm = (Mat_LMVM *)B->data;
  Mat_DQN  *lqn  = (Mat_DQN *)lmvm->ctx;
  PetscBool is_dbfgs, is_ddfp, is_dqn;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)B, MATLMVMDBFGS, &is_dbfgs));
  PetscCall(PetscObjectTypeCompare((PetscObject)B, MATLMVMDDFP, &is_ddfp));
  PetscCall(PetscObjectTypeCompare((PetscObject)B, MATLMVMDQN, &is_dqn));
  PetscCall(MatSetFromOptions_LMVM(B, PetscOptionsObject));
  PetscOptionsHeadBegin(PetscOptionsObject, "Dense symmetric Broyden method for approximating SPD Jacobian actions");
  if (is_dqn) {
    PetscCall(PetscOptionsEnum("-mat_lqn_type", "Implementation options for L-QN", "MatLMVMDenseType", MatLMVMDenseTypes, (PetscEnum)lqn->strategy, (PetscEnum *)&lqn->strategy, NULL));
  } else if (is_dbfgs) {
    PetscCall(PetscOptionsBool("-mat_lbfgs_recursive", "Use recursive formulation for MatMult_LMVMDBFGS, instead of Cholesky", "", lqn->use_recursive, &lqn->use_recursive, NULL));
    PetscCall(PetscOptionsEnum("-mat_lbfgs_type", "Implementation options for L-BFGS", "MatLMVMDenseType", MatLMVMDenseTypes, (PetscEnum)lqn->strategy, (PetscEnum *)&lqn->strategy, NULL));
  } else if (is_ddfp) {
    PetscCall(PetscOptionsBool("-mat_ldfp_recursive", "Use recursive formulation for MatSolve_LMVMDDFP, instead of Cholesky", "", lqn->use_recursive, &lqn->use_recursive, NULL));
    PetscCall(PetscOptionsEnum("-mat_ldfp_type", "Implementation options for L-DFP", "MatLMVMDenseType", MatLMVMDenseTypes, (PetscEnum)lqn->strategy, (PetscEnum *)&lqn->strategy, NULL));
  } else {
    SETERRQ(PetscObjectComm((PetscObject)B), PETSC_ERR_ARG_INCOMP, "MatSetFromOptions_LMVMDQN is only available for dense derived types. (DBFGS, DDFP, DQN");
  }
  PetscCall(SymBroydenRescaleSetFromOptions(B, lqn->rescale, PetscOptionsObject));
  PetscOptionsHeadEnd();
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatDestroy_LMVMDQN(Mat B)
{
  Mat_LMVM *lmvm = (Mat_LMVM *)B->data;
  Mat_DQN  *lqn  = (Mat_DQN *)lmvm->ctx;

  PetscFunctionBegin;
  PetscCall(SymBroydenRescaleDestroy(&lqn->rescale));
  PetscCall(MatReset_LMVMDQN_Internal(B, MAT_LMVM_RESET_ALL));
  PetscCall(PetscFree(lqn->workscalar));
  PetscCall(PetscFree(lmvm->ctx));
  PetscCall(MatDestroy_LMVM(B));
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatLMVMSymBroydenSetDelta_C", NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatUpdate_LMVMDQN(Mat B, Vec X, Vec F)
{
  Mat_LMVM *lmvm  = (Mat_LMVM *)B->data;
  Mat_DQN  *lqn   = (Mat_DQN *)lmvm->ctx;
  Mat       Sfull = lmvm->basis[LMBASIS_S]->vecs;
  Mat       Yfull = lmvm->basis[LMBASIS_Y]->vecs;

  PetscBool          is_ddfp, is_dbfgs, is_dqn;
  PetscDeviceContext dctx;

  PetscFunctionBegin;
  if (!lmvm->m) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(PetscObjectTypeCompare((PetscObject)B, MATLMVMDBFGS, &is_dbfgs));
  PetscCall(PetscObjectTypeCompare((PetscObject)B, MATLMVMDDFP, &is_ddfp));
  PetscCall(PetscObjectTypeCompare((PetscObject)B, MATLMVMDQN, &is_dqn));
  PetscCall(PetscDeviceContextGetCurrentContext(&dctx));
  if (lmvm->prev_set) {
    Vec         FX[2];
    PetscScalar dotFX[2];
    PetscScalar stFprev;
    PetscScalar curvature, yTy;
    PetscReal   curvtol;

    /* Compute the new (S = X - Xprev) and (Y = F - Fprev) vectors */
    PetscCall(VecAYPX(lmvm->Xprev, -1.0, X));
    /* Test if the updates can be accepted */
    FX[0] = lmvm->Fprev; /* dotFX[0] = s^T Fprev */
    FX[1] = F;           /* dotFX[1] = s^T F     */
    PetscCall(VecMDot(lmvm->Xprev, 2, FX, dotFX));
    PetscCall(VecAYPX(lmvm->Fprev, -1.0, F));
    PetscCall(VecDot(lmvm->Fprev, lmvm->Fprev, &yTy));
    stFprev   = PetscConj(dotFX[0]);
    curvature = PetscConj(dotFX[1] - dotFX[0]); /* s^T y */
    if (PetscRealPart(yTy) < lmvm->eps) {
      curvtol = 0.0;
    } else {
      curvtol = lmvm->eps * PetscRealPart(yTy);
    }
    if (PetscRealPart(curvature) > curvtol) {
      PetscInt m     = lmvm->m;
      PetscInt k     = lmvm->k;
      PetscInt h_old = k - oldest_update(m, k);
      PetscInt h_new = k + 1 - oldest_update(m, k + 1);
      PetscInt idx   = recycle_index(m, k);

      /* Update is good, accept it */
      PetscCall(MatUpdateKernel_LMVM(B, lmvm->Xprev, lmvm->Fprev));
      lqn->num_updates++;
      lqn->watchdog = 0;
      lqn->needPQ   = PETSC_TRUE;

      if (h_old == m && lqn->strategy == MAT_LMVM_DENSE_REORDER) {
        if (is_dqn) {
          PetscCall(MatMove_LR3(B, lqn->StY_triu, m - 1));
          PetscCall(MatMove_LR3(B, lqn->YtS_triu, m - 1));
        } else if (is_dbfgs) {
          PetscCall(MatMove_LR3(B, lqn->StY_triu, m - 1));
        } else if (is_ddfp) {
          PetscCall(MatMove_LR3(B, lqn->YtS_triu, m - 1));
        } else {
          SETERRQ(PetscObjectComm((PetscObject)B), PETSC_ERR_ARG_INCOMP, "MatUpdate_LMVMDQN is only available for dense derived types. (DBFGS, DDFP, DQN");
        }
      }

      if (lqn->use_recursive && (is_dbfgs || is_ddfp)) lqn->yts[idx] = PetscRealPart(curvature);

      if (is_dqn || is_dbfgs) { /* implement the scheme of Byrd, Nocedal, and Schnabel to save a MatMultTranspose call in the common case the       *
         * H_k is immediately applied to F after begin updated.   The S^T y computation can be split up as S^T (F - F_prev) */
        PetscInt     local_n;
        PetscScalar *StFprev;
        PetscMemType memtype;
        PetscInt     StYidx;

        StYidx = (lqn->strategy == MAT_LMVM_DENSE_REORDER) ? history_index(m, lqn->num_updates, k) : idx;
        if (!lqn->StFprev) PetscCall(VecDuplicate(lqn->rwork1, &lqn->StFprev));
        PetscCall(VecGetLocalSize(lqn->StFprev, &local_n));
        PetscCall(VecGetArrayAndMemType(lqn->StFprev, &StFprev, &memtype));
        if (local_n) {
          if (PetscMemTypeHost(memtype)) {
            StFprev[idx] = stFprev;
          } else {
            PetscCall(PetscDeviceRegisterMemory(&stFprev, PETSC_MEMTYPE_HOST, 1 * sizeof(stFprev)));
            PetscCall(PetscDeviceRegisterMemory(StFprev, memtype, local_n * sizeof(*StFprev)));
            PetscCall(PetscDeviceArrayCopy(dctx, &StFprev[idx], &stFprev, 1));
          }
        }
        PetscCall(VecRestoreArrayAndMemType(lqn->StFprev, &StFprev));

        {
          Vec this_sy_col;
          /* Now StFprev is updated for the new S vector.  Write -StFprev into the appropriate row */
          PetscCall(MatDenseGetColumnVecWrite(lqn->StY_triu, StYidx, &this_sy_col));
          PetscCall(VecAXPBY(this_sy_col, -1.0, 0.0, lqn->StFprev));

          /* Now compute the new StFprev */
          PetscCall(MatMultHermitianTransposeColumnRange(Sfull, F, lqn->StFprev, 0, h_new));
          lqn->St_count++;

          /* Now add StFprev: this_sy_col == S^T (F - Fprev) == S^T y */
          PetscCall(VecAXPY(this_sy_col, 1.0, lqn->StFprev));

          if (lqn->strategy == MAT_LMVM_DENSE_REORDER) PetscCall(VecRecycleOrderToHistoryOrder(B, this_sy_col, lqn->num_updates, lqn->cyclic_work_vec));
          PetscCall(MatDenseRestoreColumnVecWrite(lqn->StY_triu, StYidx, &this_sy_col));
        }
      }

      if (is_ddfp || is_dqn) {
        PetscInt YtSidx;

        YtSidx = (lqn->strategy == MAT_LMVM_DENSE_REORDER) ? history_index(m, lqn->num_updates, k) : idx;

        {
          Vec this_ys_col;

          PetscCall(MatDenseGetColumnVecWrite(lqn->YtS_triu, YtSidx, &this_ys_col));
          PetscCall(MatMultHermitianTransposeColumnRange(Yfull, lmvm->Xprev, this_ys_col, 0, h_new));
          lqn->Yt_count++;

          if (lqn->strategy == MAT_LMVM_DENSE_REORDER) PetscCall(VecRecycleOrderToHistoryOrder(B, this_ys_col, lqn->num_updates, lqn->cyclic_work_vec));
          PetscCall(MatDenseRestoreColumnVecWrite(lqn->YtS_triu, YtSidx, &this_ys_col));
        }
      }

      if (is_dbfgs || is_dqn) {
        PetscCall(MatGetDiagonal(lqn->StY_triu, lqn->diag_vec));
      } else if (is_ddfp) {
        PetscCall(MatGetDiagonal(lqn->YtS_triu, lqn->diag_vec));
      } else {
        SETERRQ(PetscObjectComm((PetscObject)B), PETSC_ERR_ARG_INCOMP, "MatUpdate_LMVMDQN is only available for dense derived types. (DBFGS, DDFP, DQN");
      }

      if (lqn->strategy == MAT_LMVM_DENSE_REORDER) {
        if (!lqn->diag_vec_recycle_order) PetscCall(VecDuplicate(lqn->diag_vec, &lqn->diag_vec_recycle_order));
        PetscCall(VecCopy(lqn->diag_vec, lqn->diag_vec_recycle_order));
        PetscCall(VecHistoryOrderToRecycleOrder(B, lqn->diag_vec_recycle_order, lqn->num_updates, lqn->cyclic_work_vec));
      } else {
        if (!lqn->diag_vec_recycle_order) {
          PetscCall(PetscObjectReference((PetscObject)lqn->diag_vec));
          lqn->diag_vec_recycle_order = lqn->diag_vec;
        }
      }

      PetscCall(SymBroydenRescaleUpdate(B, lqn->rescale));
    } else {
      /* Update is bad, skip it */
      ++lmvm->nrejects;
      ++lqn->watchdog;
      PetscInt m = lmvm->m;
      PetscInt k = lmvm->k;
      PetscInt h = k - oldest_update(m, k);

      /* we still have to maintain StFprev */
      if (!lqn->StFprev) PetscCall(VecDuplicate(lqn->rwork1, &lqn->StFprev));
      PetscCall(MatMultHermitianTransposeColumnRange(Sfull, F, lqn->StFprev, 0, h));
      lqn->St_count++;
    }
  }

  if (lqn->watchdog > lqn->max_seq_rejects) PetscCall(MatLMVMReset(B, PETSC_FALSE));

  /* Save the solution and function to be used in the next update */
  PetscCall(VecCopy(X, lmvm->Xprev));
  PetscCall(VecCopy(F, lmvm->Fprev));
  PetscCall(PetscObjectReference((PetscObject)F));
  PetscCall(VecDestroy(&lqn->Fprev_ref));
  lqn->Fprev_ref = F;
  PetscCall(PetscObjectStateGet((PetscObject)F, &lqn->Fprev_state));
  lmvm->prev_set = PETSC_TRUE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatDestroyThenCopy(Mat src, Mat *dst)
{
  PetscFunctionBegin;
  PetscCall(MatDestroy(dst));
  if (src) PetscCall(MatDuplicate(src, MAT_COPY_VALUES, dst));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode VecDestroyThenCopy(Vec src, Vec *dst)
{
  PetscFunctionBegin;
  PetscCall(VecDestroy(dst));
  if (src) {
    PetscCall(VecDuplicate(src, dst));
    PetscCall(VecCopy(src, *dst));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatCopy_LMVMDQN(Mat B, Mat M, MatStructure str)
{
  Mat_LMVM *bdata = (Mat_LMVM *)B->data;
  Mat_DQN  *blqn  = (Mat_DQN *)bdata->ctx;
  Mat_LMVM *mdata = (Mat_LMVM *)M->data;
  Mat_DQN  *mlqn  = (Mat_DQN *)mdata->ctx;
  PetscInt  i;
  PetscBool is_dbfgs, is_ddfp, is_dqn;

  PetscFunctionBegin;
  PetscCall(SymBroydenRescaleCopy(blqn->rescale, mlqn->rescale));
  mlqn->num_updates      = blqn->num_updates;
  mlqn->num_mult_updates = blqn->num_mult_updates;
  mlqn->dense_type       = blqn->dense_type;
  mlqn->strategy         = blqn->strategy;
  mlqn->S_count          = 0;
  mlqn->St_count         = 0;
  mlqn->Y_count          = 0;
  mlqn->Yt_count         = 0;
  mlqn->watchdog         = blqn->watchdog;
  mlqn->max_seq_rejects  = blqn->max_seq_rejects;
  mlqn->use_recursive    = blqn->use_recursive;
  mlqn->needPQ           = blqn->needPQ;
  if (blqn->allocated) {
    PetscCall(MatAllocate_LMVMDQN_Internal(M));
    PetscCall(PetscObjectTypeCompare((PetscObject)B, MATLMVMDBFGS, &is_dbfgs));
    PetscCall(PetscObjectTypeCompare((PetscObject)B, MATLMVMDDFP, &is_ddfp));
    PetscCall(PetscObjectTypeCompare((PetscObject)B, MATLMVMDQN, &is_dqn));
    PetscCall(MatDestroyThenCopy(blqn->HY, &mlqn->BS));
    PetscCall(VecDestroyThenCopy(blqn->StFprev, &mlqn->StFprev));
    PetscCall(MatDestroyThenCopy(blqn->StY_triu, &mlqn->StY_triu));
    PetscCall(MatDestroyThenCopy(blqn->StY_triu_strict, &mlqn->StY_triu_strict));
    PetscCall(MatDestroyThenCopy(blqn->YtS_triu, &mlqn->YtS_triu));
    PetscCall(MatDestroyThenCopy(blqn->YtS_triu_strict, &mlqn->YtS_triu_strict));
    PetscCall(MatDestroyThenCopy(blqn->YtHY, &mlqn->YtHY));
    PetscCall(MatDestroyThenCopy(blqn->StBS, &mlqn->StBS));
    PetscCall(MatDestroyThenCopy(blqn->J, &mlqn->J));
    PetscCall(VecDestroyThenCopy(blqn->diag_vec, &mlqn->diag_vec));
    PetscCall(VecDestroyThenCopy(blqn->diag_vec_recycle_order, &mlqn->diag_vec_recycle_order));
    PetscCall(VecDestroyThenCopy(blqn->inv_diag_vec, &mlqn->inv_diag_vec));
    if (blqn->use_recursive && (is_dbfgs || is_ddfp)) {
      for (i = 0; i < bdata->m; i++) {
        PetscCall(VecDestroyThenCopy(blqn->PQ[i], &mlqn->PQ[i]));
        mlqn->yts[i] = blqn->yts[i];
        if (is_dbfgs) {
          mlqn->stp[i] = blqn->stp[i];
        } else if (is_ddfp) {
          mlqn->ytq[i] = blqn->ytq[i];
        }
      }
    }
  }
  PetscCall(PetscObjectReference((PetscObject)blqn->Fprev_ref));
  PetscCall(VecDestroy(&mlqn->Fprev_ref));
  mlqn->Fprev_ref   = blqn->Fprev_ref;
  mlqn->Fprev_state = blqn->Fprev_state;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatMult_LMVMDQN(Mat B, Vec X, Vec Z)
{
  PetscFunctionBegin;
  PetscCall(MatMult_LMVMDDFP(B, X, Z));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatSolve_LMVMDQN(Mat H, Vec F, Vec dX)
{
  PetscFunctionBegin;
  PetscCall(MatSolve_LMVMDBFGS(H, F, dX));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatLMVMSymBroydenSetDelta_LMVMDQN(Mat B, PetscScalar delta)
{
  Mat_LMVM *lmvm = (Mat_LMVM *)B->data;
  Mat_DQN  *lqn  = (Mat_DQN *)lmvm->ctx;

  PetscFunctionBegin;
  PetscCall(SymBroydenRescaleSetDelta(B, lqn->rescale, PetscAbsReal(PetscRealPart(delta))));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  This dense representation uses Davidon-Fletcher-Powell (DFP) for MatMult,
  and Broyden-Fletcher-Goldfarb-Shanno (BFGS) for MatSolve. This implementation
  results in avoiding costly Cholesky factorization, at the cost of duality cap.
  Please refer to MatLMVMDDFP and MatLMVMDBFGS for more information.
*/
PetscErrorCode MatCreate_LMVMDQN(Mat B)
{
  Mat_LMVM *lmvm;
  Mat_DQN  *lqn;

  PetscFunctionBegin;
  PetscCall(MatCreate_LMVM(B));
  PetscCall(PetscObjectChangeTypeName((PetscObject)B, MATLMVMDQN));
  PetscCall(MatSetOption(B, MAT_HERMITIAN, PETSC_TRUE));
  PetscCall(MatSetOption(B, MAT_SPD, PETSC_TRUE));
  PetscCall(MatSetOption(B, MAT_SPD_ETERNAL, PETSC_TRUE));
  B->ops->view           = MatView_LMVMDQN;
  B->ops->setup          = MatSetUp_LMVMDQN;
  B->ops->setfromoptions = MatSetFromOptions_LMVMDQN;
  B->ops->destroy        = MatDestroy_LMVMDQN;

  lmvm              = (Mat_LMVM *)B->data;
  lmvm->ops->reset  = MatReset_LMVMDQN;
  lmvm->ops->update = MatUpdate_LMVMDQN;
  lmvm->ops->mult   = MatMult_LMVMDQN;
  lmvm->ops->solve  = MatSolve_LMVMDQN;
  lmvm->ops->copy   = MatCopy_LMVMDQN;

  lmvm->ops->multht  = lmvm->ops->mult;
  lmvm->ops->solveht = lmvm->ops->solve;

  PetscCall(PetscNew(&lqn));
  lmvm->ctx            = (void *)lqn;
  lqn->allocated       = PETSC_FALSE;
  lqn->use_recursive   = PETSC_FALSE;
  lqn->needPQ          = PETSC_FALSE;
  lqn->watchdog        = 0;
  lqn->max_seq_rejects = lmvm->m / 2;
  lqn->strategy        = MAT_LMVM_DENSE_INPLACE;

  PetscCall(SymBroydenRescaleCreate(&lqn->rescale));
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatLMVMSymBroydenSetDelta_C", MatLMVMSymBroydenSetDelta_LMVMDQN));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  MatCreateLMVMDQN - Creates a dense representation of the limited-memory
  Quasi-Newton approximation to a Hessian.

  Collective

  Input Parameters:
+ comm - MPI communicator
. n    - number of local rows for storage vectors
- N    - global size of the storage vectors

  Output Parameter:
. B - the matrix

  Level: advanced

  Note:
  It is recommended that one use the `MatCreate()`, `MatSetType()` and/or `MatSetFromOptions()`
  paradigm instead of this routine directly.

.seealso: `MatCreate()`, `MATLMVM`, `MATLMVMDBFGS`, `MATLMVMDDFP`, `MatCreateLMVMDDFP()`, `MatCreateLMVMDBFGS()`
@*/
PetscErrorCode MatCreateLMVMDQN(MPI_Comm comm, PetscInt n, PetscInt N, Mat *B)
{
  PetscFunctionBegin;
  PetscCall(KSPInitializePackage());
  PetscCall(MatCreate(comm, B));
  PetscCall(MatSetSizes(*B, n, n, N, N));
  PetscCall(MatSetType(*B, MATLMVMDQN));
  PetscCall(MatSetUp(*B));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatDQNApplyJ0Fwd(Mat B, Vec X, Vec Z)
{
  PetscFunctionBegin;
  PetscCall(MatLMVMApplyJ0Fwd(B, X, Z));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatDQNApplyJ0Inv(Mat B, Vec F, Vec dX)
{
  PetscFunctionBegin;
  PetscCall(MatLMVMApplyJ0Inv(B, F, dX));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* This is not Bunch-Kaufman LDLT: here L is strictly lower triangular part of STY */
static PetscErrorCode MatGetLDLT(Mat B, Mat result)
{
  Mat_LMVM *lmvm  = (Mat_LMVM *)B->data;
  Mat_DQN  *lbfgs = (Mat_DQN *)lmvm->ctx;
  PetscInt  m_local;

  PetscFunctionBegin;
  if (!lbfgs->temp_mat) PetscCall(MatDuplicate(lbfgs->YtS_triu_strict, MAT_SHARE_NONZERO_PATTERN, &lbfgs->temp_mat));
  PetscCall(MatCopy(lbfgs->YtS_triu_strict, lbfgs->temp_mat, SAME_NONZERO_PATTERN));
  PetscCall(MatDiagonalScale(lbfgs->temp_mat, lbfgs->inv_diag_vec, NULL));
  PetscCall(MatGetLocalSize(result, &m_local, NULL));
  // need to conjugate and conjugate again because we have MatTransposeMatMult but not MatHermitianTransposeMatMult()
  PetscCall(MatConjugate(lbfgs->temp_mat));
  if (m_local) {
    Mat temp_local, YtS_local, result_local;
    PetscCall(MatDenseGetLocalMatrix(lbfgs->YtS_triu_strict, &YtS_local));
    PetscCall(MatDenseGetLocalMatrix(lbfgs->temp_mat, &temp_local));
    PetscCall(MatDenseGetLocalMatrix(result, &result_local));
    PetscCall(MatTransposeMatMult(YtS_local, temp_local, MAT_REUSE_MATRIX, PETSC_DETERMINE, &result_local));
  }
  PetscCall(MatConjugate(result));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatLMVMDBFGSUpdateMultData(Mat B)
{
  Mat_LMVM *lmvm  = (Mat_LMVM *)B->data;
  Mat_DQN  *lbfgs = (Mat_DQN *)lmvm->ctx;
  PetscInt  m     = lmvm->m, m_local;
  PetscInt  k     = lmvm->k;
  PetscInt  h     = k - oldest_update(m, k);
  PetscInt  j_0;
  PetscInt  prev_oldest;
  Mat       J_local;
  Mat       Sfull = lmvm->basis[LMBASIS_S]->vecs;
  Mat       Yfull = lmvm->basis[LMBASIS_Y]->vecs;

  PetscFunctionBegin;
  if (!lbfgs->YtS_triu_strict) {
    PetscCall(MatDuplicate(lbfgs->StY_triu, MAT_SHARE_NONZERO_PATTERN, &lbfgs->YtS_triu_strict));
    PetscCall(MatDestroy(&lbfgs->StBS));
    PetscCall(MatDuplicate(lbfgs->StY_triu, MAT_SHARE_NONZERO_PATTERN, &lbfgs->StBS));
    PetscCall(MatDestroy(&lbfgs->J));
    PetscCall(MatDuplicate(lbfgs->StY_triu, MAT_SHARE_NONZERO_PATTERN, &lbfgs->J));
    PetscCall(MatDestroy(&lbfgs->BS));
    PetscCall(MatDuplicate(Yfull, MAT_SHARE_NONZERO_PATTERN, &lbfgs->BS));
    PetscCall(MatShift(lbfgs->StBS, 1.0));
    lbfgs->num_mult_updates = oldest_update(m, k);
  }
  if (lbfgs->num_mult_updates == k) PetscFunctionReturn(PETSC_SUCCESS);

  /* B_0 may have been updated, we must recompute B_0 S and S^T B_0 S */
  for (PetscInt j = oldest_update(m, k); j < k; j++) {
    Vec      s_j;
    Vec      Bs_j;
    Vec      StBs_j;
    PetscInt S_idx    = recycle_index(m, j);
    PetscInt StBS_idx = lbfgs->strategy == MAT_LMVM_DENSE_INPLACE ? S_idx : history_index(m, k, j);

    PetscCall(MatDenseGetColumnVecWrite(lbfgs->BS, S_idx, &Bs_j));
    PetscCall(MatDenseGetColumnVecRead(Sfull, S_idx, &s_j));
    PetscCall(MatDQNApplyJ0Fwd(B, s_j, Bs_j));
    PetscCall(MatDenseRestoreColumnVecRead(Sfull, S_idx, &s_j));
    PetscCall(MatDenseGetColumnVecWrite(lbfgs->StBS, StBS_idx, &StBs_j));
    PetscCall(MatMultHermitianTransposeColumnRange(Sfull, Bs_j, StBs_j, 0, h));
    lbfgs->St_count++;
    if (lbfgs->strategy == MAT_LMVM_DENSE_REORDER) PetscCall(VecRecycleOrderToHistoryOrder(B, StBs_j, lbfgs->num_updates, lbfgs->cyclic_work_vec));
    PetscCall(MatDenseRestoreColumnVecWrite(lbfgs->StBS, StBS_idx, &StBs_j));
    PetscCall(MatDenseRestoreColumnVecWrite(lbfgs->BS, S_idx, &Bs_j));
  }
  prev_oldest = oldest_update(m, lbfgs->num_mult_updates);
  if (lbfgs->strategy == MAT_LMVM_DENSE_REORDER && prev_oldest < oldest_update(m, k)) {
    /* move the YtS entries that have been computed and need to be kept back up */
    PetscInt m_keep = m - (oldest_update(m, k) - prev_oldest);

    PetscCall(MatMove_LR3(B, lbfgs->YtS_triu_strict, m_keep));
  }
  PetscCall(MatGetLocalSize(lbfgs->YtS_triu_strict, &m_local, NULL));
  j_0 = PetscMax(lbfgs->num_mult_updates, oldest_update(m, k));
  for (PetscInt j = j_0; j < k; j++) {
    PetscInt S_idx   = recycle_index(m, j);
    PetscInt YtS_idx = lbfgs->strategy == MAT_LMVM_DENSE_INPLACE ? S_idx : history_index(m, k, j);
    Vec      s_j, Yts_j;

    PetscCall(MatDenseGetColumnVecRead(Sfull, S_idx, &s_j));
    PetscCall(MatDenseGetColumnVecWrite(lbfgs->YtS_triu_strict, YtS_idx, &Yts_j));
    PetscCall(MatMultHermitianTransposeColumnRange(Yfull, s_j, Yts_j, 0, h));
    lbfgs->Yt_count++;
    if (lbfgs->strategy == MAT_LMVM_DENSE_REORDER) PetscCall(VecRecycleOrderToHistoryOrder(B, Yts_j, lbfgs->num_updates, lbfgs->cyclic_work_vec));
    PetscCall(MatDenseRestoreColumnVecWrite(lbfgs->YtS_triu_strict, YtS_idx, &Yts_j));
    PetscCall(MatDenseRestoreColumnVecRead(Sfull, S_idx, &s_j));
    /* zero the corresponding row */
    if (m_local > 0) {
      Mat YtS_local, YtS_row;

      PetscCall(MatDenseGetLocalMatrix(lbfgs->YtS_triu_strict, &YtS_local));
      PetscCall(MatDenseGetSubMatrix(YtS_local, YtS_idx, YtS_idx + 1, PETSC_DECIDE, PETSC_DECIDE, &YtS_row));
      PetscCall(MatZeroEntries(YtS_row));
      PetscCall(MatDenseRestoreSubMatrix(YtS_local, &YtS_row));
    }
  }
  if (!lbfgs->inv_diag_vec) PetscCall(VecDuplicate(lbfgs->diag_vec, &lbfgs->inv_diag_vec));
  PetscCall(VecCopy(lbfgs->diag_vec, lbfgs->inv_diag_vec));
  PetscCall(VecReciprocal(lbfgs->inv_diag_vec));
  PetscCall(MatDenseGetLocalMatrix(lbfgs->J, &J_local));
  PetscCall(MatSetFactorType(J_local, MAT_FACTOR_NONE));
  PetscCall(MatGetLDLT(B, lbfgs->J));
  PetscCall(MatAXPY(lbfgs->J, 1.0, lbfgs->StBS, SAME_NONZERO_PATTERN));
  if (m_local) {
    PetscCall(MatSetOption(J_local, MAT_SPD, PETSC_TRUE));
    PetscCall(MatCholeskyFactor(J_local, NULL, NULL));
  }
  lbfgs->num_mult_updates = lbfgs->num_updates;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Solves for
 * [ I | -S R^{-T} ] [  I  | 0 ] [ H_0 | 0 ] [ I | Y ] [      I      ]
 *                   [-----+---] [-----+---] [---+---] [-------------]
 *                   [ Y^T | I ] [  0  | D ] [ 0 | I ] [ -R^{-1} S^T ]  */

static PetscErrorCode MatSolve_LMVMDBFGS(Mat H, Vec F, Vec dX)
{
  Mat_LMVM        *lmvm   = (Mat_LMVM *)H->data;
  Mat_DQN         *lbfgs  = (Mat_DQN *)lmvm->ctx;
  Vec              rwork1 = lbfgs->rwork1;
  PetscInt         m      = lmvm->m;
  PetscInt         k      = lmvm->k;
  PetscInt         h      = k - oldest_update(m, k);
  Mat              Sfull  = lmvm->basis[LMBASIS_S]->vecs;
  Mat              Yfull  = lmvm->basis[LMBASIS_Y]->vecs;
  PetscObjectState Fstate;

  PetscFunctionBegin;
  VecCheckSameSize(F, 2, dX, 3);
  VecCheckMatCompatible(H, dX, 3, F, 2);

  /* Block Version */
  if (!lbfgs->num_updates) {
    PetscCall(MatDQNApplyJ0Inv(H, F, dX));
    PetscFunctionReturn(PETSC_SUCCESS); /* No updates stored yet */
  }

  PetscCall(PetscObjectStateGet((PetscObject)F, &Fstate));
  if (F == lbfgs->Fprev_ref && Fstate == lbfgs->Fprev_state) {
    PetscCall(VecCopy(lbfgs->StFprev, rwork1));
  } else {
    PetscCall(MatMultHermitianTransposeColumnRange(Sfull, F, rwork1, 0, h));
    lbfgs->St_count++;
  }

  /* Reordering rwork1, as STY is in history order, while S is in recycled order */
  if (lbfgs->strategy == MAT_LMVM_DENSE_REORDER) PetscCall(VecRecycleOrderToHistoryOrder(H, rwork1, lbfgs->num_updates, lbfgs->cyclic_work_vec));
  PetscCall(MatUpperTriangularSolveInPlace(H, lbfgs->StY_triu, rwork1, PETSC_FALSE, lbfgs->num_updates, lbfgs->strategy));
  PetscCall(VecScale(rwork1, -1.0));
  if (lbfgs->strategy == MAT_LMVM_DENSE_REORDER) PetscCall(VecHistoryOrderToRecycleOrder(H, rwork1, lbfgs->num_updates, lbfgs->cyclic_work_vec));

  PetscCall(VecCopy(F, lbfgs->column_work));
  PetscCall(MatMultAddColumnRange(Yfull, rwork1, lbfgs->column_work, lbfgs->column_work, 0, h));
  lbfgs->Y_count++;

  PetscCall(VecPointwiseMult(rwork1, lbfgs->diag_vec_recycle_order, rwork1));
  PetscCall(MatDQNApplyJ0Inv(H, lbfgs->column_work, dX));

  PetscCall(MatMultHermitianTransposeAddColumnRange(Yfull, dX, rwork1, rwork1, 0, h));
  lbfgs->Yt_count++;

  if (lbfgs->strategy == MAT_LMVM_DENSE_REORDER) PetscCall(VecRecycleOrderToHistoryOrder(H, rwork1, lbfgs->num_updates, lbfgs->cyclic_work_vec));
  PetscCall(MatUpperTriangularSolveInPlace(H, lbfgs->StY_triu, rwork1, PETSC_TRUE, lbfgs->num_updates, lbfgs->strategy));
  PetscCall(VecScale(rwork1, -1.0));
  if (lbfgs->strategy == MAT_LMVM_DENSE_REORDER) PetscCall(VecHistoryOrderToRecycleOrder(H, rwork1, lbfgs->num_updates, lbfgs->cyclic_work_vec));

  PetscCall(MatMultAddColumnRange(Sfull, rwork1, dX, dX, 0, h));
  lbfgs->S_count++;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Solves for
   B_0 - [ Y | B_0 S] [ -D  |    L^T    ]^-1 [   Y^T   ]
                      [-----+-----------]    [---------]
                      [  L  | S^T B_0 S ]    [ S^T B_0 ]

   Above is equivalent to

   B_0 - [ Y | B_0 S] [[     I     | 0 ][ -D  | 0 ][ I | -D^{-1} L^T ]]^-1 [   Y^T   ]
                      [[-----------+---][-----+---][---+-------------]]    [---------]
                      [[ -L D^{-1} | I ][  0  | J ][ 0 |       I     ]]    [ S^T B_0 ]

   where J = S^T B_0 S + L D^{-1} L^T

   becomes

   B_0 - [ Y | B_0 S] [ I | D^{-1} L^T ][ -D^{-1}  |   0    ][    I     | 0 ] [   Y^T   ]
                      [---+------------][----------+--------][----------+---] [---------]
                      [ 0 |     I      ][     0    | J^{-1} ][ L D^{-1} | I ] [ S^T B_0 ]

                      =

   B_0 + [ Y | B_0 S] [ D^{-1} | 0 ][ I | L^T ][ I |    0    ][     I    | 0 ] [   Y^T   ]
                      [--------+---][---+-----][---+---------][----------+---] [---------]
                      [ 0      | I ][ 0 |  I  ][ 0 | -J^{-1} ][ L D^{-1} | I ] [ S^T B_0 ]

                      (Note that YtS_triu_strict is L^T)
   Byrd, Nocedal, Schnabel 1994

   Alternative approach: considering the fact that DFP is dual to BFGS, use MatMult of DPF:
   (See ddfp.c's MatMult_LMVMDDFP)

*/
static PetscErrorCode MatMult_LMVMDBFGS(Mat B, Vec X, Vec Z)
{
  Mat_LMVM *lmvm  = (Mat_LMVM *)B->data;
  Mat_DQN  *lbfgs = (Mat_DQN *)lmvm->ctx;
  Mat       J_local;
  PetscInt  idx, i, j, m_local, local_n;
  PetscInt  m     = lmvm->m;
  PetscInt  k     = lmvm->k;
  PetscInt  h     = k - oldest_update(m, k);
  Mat       Sfull = lmvm->basis[LMBASIS_S]->vecs;
  Mat       Yfull = lmvm->basis[LMBASIS_Y]->vecs;

  PetscFunctionBegin;
  VecCheckSameSize(X, 2, Z, 3);
  VecCheckMatCompatible(B, X, 2, Z, 3);

  /* Cholesky Version */
  /* Start with the B0 term */
  PetscCall(MatDQNApplyJ0Fwd(B, X, Z));
  if (!lbfgs->num_updates) PetscFunctionReturn(PETSC_SUCCESS); /* No updates stored yet */

  if (lbfgs->use_recursive) {
    PetscDeviceContext dctx;
    PetscMemType       memtype;
    PetscScalar        stz, ytx, stp, sjtpi, yjtsi, *workscalar;
    PetscInt           oldest = oldest_update(m, k);

    PetscCall(PetscDeviceContextGetCurrentContext(&dctx));
    /* Recursive formulation to avoid Cholesky. Not a dense formulation */
    PetscCall(MatMultHermitianTransposeColumnRange(Yfull, X, lbfgs->rwork1, 0, h));
    lbfgs->Yt_count++;

    PetscCall(VecGetLocalSize(lbfgs->rwork1, &local_n));

    if (lbfgs->needPQ) {
      PetscInt oldest = oldest_update(m, k);
      for (i = oldest; i < k; ++i) {
        idx = recycle_index(m, i);
        /* column_work = S[idx] */
        PetscCall(MatGetColumnVector(Sfull, lbfgs->column_work, idx));
        PetscCall(MatDQNApplyJ0Fwd(B, lbfgs->column_work, lbfgs->PQ[idx]));
        PetscCall(MatMultHermitianTransposeColumnRange(Yfull, lbfgs->column_work, lbfgs->rwork3, 0, h));
        PetscCall(VecGetArrayAndMemType(lbfgs->rwork3, &workscalar, &memtype));
        for (j = oldest; j < i; ++j) {
          PetscInt idx_j = recycle_index(m, j);
          /* Copy yjtsi in device-aware manner */
          if (local_n) {
            if (PetscMemTypeHost(memtype)) {
              yjtsi = workscalar[idx_j];
            } else {
              PetscCall(PetscDeviceRegisterMemory(&yjtsi, PETSC_MEMTYPE_HOST, sizeof(yjtsi)));
              PetscCall(PetscDeviceRegisterMemory(workscalar, memtype, local_n * sizeof(*workscalar)));
              PetscCall(PetscDeviceArrayCopy(dctx, &yjtsi, &workscalar[idx_j], 1));
            }
          }
          PetscCallMPI(MPI_Bcast(&yjtsi, 1, MPIU_SCALAR, 0, PetscObjectComm((PetscObject)B)));
          /* column_work2 = S[j] */
          PetscCall(MatGetColumnVector(Sfull, lbfgs->column_work2, idx_j));
          PetscCall(VecDot(lbfgs->PQ[idx], lbfgs->column_work2, &sjtpi));
          /* column_work2 = Y[j] */
          PetscCall(MatGetColumnVector(Yfull, lbfgs->column_work2, idx_j));
          /* Compute the pure BFGS component of the forward product */
          PetscCall(VecAXPBYPCZ(lbfgs->PQ[idx], -sjtpi / lbfgs->stp[idx_j], yjtsi / lbfgs->yts[idx_j], 1.0, lbfgs->PQ[idx_j], lbfgs->column_work2));
        }
        PetscCall(VecDot(lbfgs->PQ[idx], lbfgs->column_work, &stp));
        lbfgs->stp[idx] = PetscRealPart(stp);
      }
      lbfgs->needPQ = PETSC_FALSE;
    }

    PetscCall(VecGetArrayAndMemType(lbfgs->rwork1, &workscalar, &memtype));
    for (i = oldest; i < k; ++i) {
      idx = recycle_index(m, i);
      /* Copy stz[i], ytx[i] in device-aware manner */
      if (local_n) {
        if (PetscMemTypeHost(memtype)) {
          ytx = workscalar[idx];
        } else {
          PetscCall(PetscDeviceRegisterMemory(&ytx, PETSC_MEMTYPE_HOST, 1 * sizeof(ytx)));
          PetscCall(PetscDeviceRegisterMemory(workscalar, memtype, local_n * sizeof(*workscalar)));
          PetscCall(PetscDeviceArrayCopy(dctx, &ytx, &workscalar[idx], 1));
        }
      }
      PetscCallMPI(MPI_Bcast(&ytx, 1, MPIU_SCALAR, 0, PetscObjectComm((PetscObject)B)));
      /* column_work : S[i], column_work2 : Y[i] */
      PetscCall(MatGetColumnVector(Sfull, lbfgs->column_work, idx));
      PetscCall(MatGetColumnVector(Yfull, lbfgs->column_work2, idx));
      PetscCall(VecDot(Z, lbfgs->column_work, &stz));
      PetscCall(VecAXPBYPCZ(Z, -stz / lbfgs->stp[idx], ytx / lbfgs->yts[idx], 1.0, lbfgs->PQ[idx], lbfgs->column_work2));
    }
    PetscCall(VecRestoreArrayAndMemType(lbfgs->rwork1, &workscalar));
  } else {
    PetscCall(MatLMVMDBFGSUpdateMultData(B));
    PetscCall(MatMultHermitianTransposeColumnRange(Yfull, X, lbfgs->rwork1, 0, h));
    lbfgs->Yt_count++;
    PetscCall(MatMultHermitianTransposeColumnRange(Sfull, Z, lbfgs->rwork2, 0, h));
    lbfgs->St_count++;
    if (lbfgs->strategy == MAT_LMVM_DENSE_REORDER) {
      PetscCall(VecRecycleOrderToHistoryOrder(B, lbfgs->rwork1, lbfgs->num_updates, lbfgs->cyclic_work_vec));
      PetscCall(VecRecycleOrderToHistoryOrder(B, lbfgs->rwork2, lbfgs->num_updates, lbfgs->cyclic_work_vec));
    }

    PetscCall(VecPointwiseMult(lbfgs->rwork1, lbfgs->rwork1, lbfgs->inv_diag_vec));
    if (PetscDefined(USE_COMPLEX)) PetscCall(MatConjugate(lbfgs->YtS_triu_strict));
    PetscCall(MatMultTransposeAdd(lbfgs->YtS_triu_strict, lbfgs->rwork1, lbfgs->rwork2, lbfgs->rwork2));
    if (PetscDefined(USE_COMPLEX)) PetscCall(MatConjugate(lbfgs->YtS_triu_strict));

    if (!lbfgs->rwork2_local) PetscCall(VecCreateLocalVector(lbfgs->rwork2, &lbfgs->rwork2_local));
    if (!lbfgs->rwork3_local) PetscCall(VecCreateLocalVector(lbfgs->rwork3, &lbfgs->rwork3_local));
    PetscCall(VecGetLocalVectorRead(lbfgs->rwork2, lbfgs->rwork2_local));
    PetscCall(VecGetLocalVector(lbfgs->rwork3, lbfgs->rwork3_local));
    PetscCall(MatDenseGetLocalMatrix(lbfgs->J, &J_local));
    PetscCall(VecGetSize(lbfgs->rwork2_local, &m_local));
    if (m_local) {
      PetscCall(MatDenseGetLocalMatrix(lbfgs->J, &J_local));
      PetscCall(MatSolve(J_local, lbfgs->rwork2_local, lbfgs->rwork3_local));
    }
    PetscCall(VecRestoreLocalVector(lbfgs->rwork3, lbfgs->rwork3_local));
    PetscCall(VecRestoreLocalVectorRead(lbfgs->rwork2, lbfgs->rwork2_local));
    PetscCall(VecScale(lbfgs->rwork3, -1.0));

    PetscCall(MatMult(lbfgs->YtS_triu_strict, lbfgs->rwork3, lbfgs->rwork2));
    PetscCall(VecPointwiseMult(lbfgs->rwork2, lbfgs->rwork2, lbfgs->inv_diag_vec));
    PetscCall(VecAXPY(lbfgs->rwork1, 1.0, lbfgs->rwork2));

    if (lbfgs->strategy == MAT_LMVM_DENSE_REORDER) {
      PetscCall(VecHistoryOrderToRecycleOrder(B, lbfgs->rwork1, lbfgs->num_updates, lbfgs->cyclic_work_vec));
      PetscCall(VecHistoryOrderToRecycleOrder(B, lbfgs->rwork3, lbfgs->num_updates, lbfgs->cyclic_work_vec));
    }

    PetscCall(MatMultAddColumnRange(Yfull, lbfgs->rwork1, Z, Z, 0, h));
    lbfgs->Y_count++;
    PetscCall(MatMultAddColumnRange(lbfgs->BS, lbfgs->rwork3, Z, Z, 0, h));
    lbfgs->S_count++;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  This dense representation reduces the L-BFGS update to a series of
  matrix-vector products with dense matrices in lieu of the conventional matrix-free
  two-loop algorithm.
*/
PetscErrorCode MatCreate_LMVMDBFGS(Mat B)
{
  Mat_LMVM *lmvm;
  Mat_DQN  *lbfgs;

  PetscFunctionBegin;
  PetscCall(MatCreate_LMVM(B));
  PetscCall(PetscObjectChangeTypeName((PetscObject)B, MATLMVMDBFGS));
  PetscCall(MatSetOption(B, MAT_HERMITIAN, PETSC_TRUE));
  PetscCall(MatSetOption(B, MAT_SPD, PETSC_TRUE));
  PetscCall(MatSetOption(B, MAT_SPD_ETERNAL, PETSC_TRUE));
  B->ops->view           = MatView_LMVMDQN;
  B->ops->setup          = MatSetUp_LMVMDQN;
  B->ops->setfromoptions = MatSetFromOptions_LMVMDQN;
  B->ops->destroy        = MatDestroy_LMVMDQN;

  lmvm              = (Mat_LMVM *)B->data;
  lmvm->ops->reset  = MatReset_LMVMDQN;
  lmvm->ops->update = MatUpdate_LMVMDQN;
  lmvm->ops->mult   = MatMult_LMVMDBFGS;
  lmvm->ops->solve  = MatSolve_LMVMDBFGS;
  lmvm->ops->copy   = MatCopy_LMVMDQN;

  lmvm->ops->multht  = lmvm->ops->mult;
  lmvm->ops->solveht = lmvm->ops->solve;

  PetscCall(PetscNew(&lbfgs));
  lmvm->ctx              = (void *)lbfgs;
  lbfgs->allocated       = PETSC_FALSE;
  lbfgs->use_recursive   = PETSC_TRUE;
  lbfgs->needPQ          = PETSC_TRUE;
  lbfgs->watchdog        = 0;
  lbfgs->max_seq_rejects = lmvm->m / 2;
  lbfgs->strategy        = MAT_LMVM_DENSE_INPLACE;

  PetscCall(SymBroydenRescaleCreate(&lbfgs->rescale));
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatLMVMSymBroydenSetDelta_C", MatLMVMSymBroydenSetDelta_LMVMDQN));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  MatCreateLMVMDBFGS - Creates a dense representation of the limited-memory
  Broyden-Fletcher-Goldfarb-Shanno (BFGS) approximation to a Hessian.

  Collective

  Input Parameters:
+ comm - MPI communicator
. n    - number of local rows for storage vectors
- N    - global size of the storage vectors

  Output Parameter:
. B - the matrix

  Level: advanced

  Note:
  It is recommended that one use the MatCreate(), MatSetType() and/or MatSetFromOptions()
  paradigm instead of this routine directly.

.seealso: `MatCreate()`, `MATLMVM`, `MATLMVMDBFGS`, `MatCreateLMVMBFGS()`
@*/
PetscErrorCode MatCreateLMVMDBFGS(MPI_Comm comm, PetscInt n, PetscInt N, Mat *B)
{
  PetscFunctionBegin;
  PetscCall(KSPInitializePackage());
  PetscCall(MatCreate(comm, B));
  PetscCall(MatSetSizes(*B, n, n, N, N));
  PetscCall(MatSetType(*B, MATLMVMDBFGS));
  PetscCall(MatSetUp(*B));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* here R is strictly upper triangular part of STY */
static PetscErrorCode MatGetRTDR(Mat B, Mat result)
{
  Mat_LMVM *lmvm = (Mat_LMVM *)B->data;
  Mat_DQN  *ldfp = (Mat_DQN *)lmvm->ctx;
  PetscInt  m_local;

  PetscFunctionBegin;
  if (!ldfp->temp_mat) PetscCall(MatDuplicate(ldfp->StY_triu_strict, MAT_SHARE_NONZERO_PATTERN, &ldfp->temp_mat));
  PetscCall(MatCopy(ldfp->StY_triu_strict, ldfp->temp_mat, SAME_NONZERO_PATTERN));
  PetscCall(MatDiagonalScale(ldfp->temp_mat, ldfp->inv_diag_vec, NULL));
  PetscCall(MatGetLocalSize(result, &m_local, NULL));
  // need to conjugate and conjugate again because we have MatTransposeMatMult but not MatHermitianTransposeMatMult()
  PetscCall(MatConjugate(ldfp->temp_mat));
  if (m_local) {
    Mat temp_local, StY_local, result_local;
    PetscCall(MatDenseGetLocalMatrix(ldfp->StY_triu_strict, &StY_local));
    PetscCall(MatDenseGetLocalMatrix(ldfp->temp_mat, &temp_local));
    PetscCall(MatDenseGetLocalMatrix(result, &result_local));
    PetscCall(MatTransposeMatMult(StY_local, temp_local, MAT_REUSE_MATRIX, PETSC_DETERMINE, &result_local));
  }
  PetscCall(MatConjugate(result));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatLMVMDDFPUpdateSolveData(Mat B)
{
  Mat_LMVM *lmvm = (Mat_LMVM *)B->data;
  Mat_DQN  *ldfp = (Mat_DQN *)lmvm->ctx;
  PetscInt  m    = lmvm->m, m_local;
  PetscInt  k    = lmvm->k;
  PetscInt  h    = k - oldest_update(m, k);
  PetscInt  j_0;
  PetscInt  prev_oldest;
  Mat       Sfull = lmvm->basis[LMBASIS_S]->vecs;
  Mat       Yfull = lmvm->basis[LMBASIS_Y]->vecs;
  Mat       J_local;

  PetscFunctionBegin;
  if (!ldfp->StY_triu_strict) {
    PetscCall(MatDuplicate(ldfp->YtS_triu, MAT_SHARE_NONZERO_PATTERN, &ldfp->StY_triu_strict));
    PetscCall(MatDestroy(&ldfp->YtHY));
    PetscCall(MatDuplicate(ldfp->YtS_triu, MAT_SHARE_NONZERO_PATTERN, &ldfp->YtHY));
    PetscCall(MatDestroy(&ldfp->J));
    PetscCall(MatDuplicate(ldfp->YtS_triu, MAT_SHARE_NONZERO_PATTERN, &ldfp->J));
    PetscCall(MatDestroy(&ldfp->HY));
    PetscCall(MatDuplicate(Yfull, MAT_SHARE_NONZERO_PATTERN, &ldfp->HY));
    PetscCall(MatShift(ldfp->YtHY, 1.0));
    ldfp->num_mult_updates = oldest_update(m, k);
  }
  if (ldfp->num_mult_updates == k) PetscFunctionReturn(PETSC_SUCCESS);

  /* H_0 may have been updated, we must recompute H_0 Y and Y^T H_0 Y */
  for (PetscInt j = oldest_update(m, k); j < k; j++) {
    Vec      y_j;
    Vec      Hy_j;
    Vec      YtHy_j;
    PetscInt Y_idx    = recycle_index(m, j);
    PetscInt YtHY_idx = ldfp->strategy == MAT_LMVM_DENSE_INPLACE ? Y_idx : history_index(m, k, j);

    PetscCall(MatDenseGetColumnVecWrite(ldfp->HY, Y_idx, &Hy_j));
    PetscCall(MatDenseGetColumnVecRead(Yfull, Y_idx, &y_j));
    PetscCall(MatDQNApplyJ0Inv(B, y_j, Hy_j));
    PetscCall(MatDenseRestoreColumnVecRead(Yfull, Y_idx, &y_j));
    PetscCall(MatDenseGetColumnVecWrite(ldfp->YtHY, YtHY_idx, &YtHy_j));
    PetscCall(MatMultHermitianTransposeColumnRange(Yfull, Hy_j, YtHy_j, 0, h));
    ldfp->Yt_count++;
    if (ldfp->strategy == MAT_LMVM_DENSE_REORDER) PetscCall(VecRecycleOrderToHistoryOrder(B, YtHy_j, ldfp->num_updates, ldfp->cyclic_work_vec));
    PetscCall(MatDenseRestoreColumnVecWrite(ldfp->YtHY, YtHY_idx, &YtHy_j));
    PetscCall(MatDenseRestoreColumnVecWrite(ldfp->HY, Y_idx, &Hy_j));
  }
  prev_oldest = oldest_update(m, ldfp->num_mult_updates);
  if (ldfp->strategy == MAT_LMVM_DENSE_REORDER && prev_oldest < oldest_update(m, k)) {
    /* move the YtS entries that have been computed and need to be kept back up */
    PetscInt m_keep = m - (oldest_update(m, k) - prev_oldest);

    PetscCall(MatMove_LR3(B, ldfp->StY_triu_strict, m_keep));
  }
  PetscCall(MatGetLocalSize(ldfp->StY_triu_strict, &m_local, NULL));
  j_0 = PetscMax(ldfp->num_mult_updates, oldest_update(m, k));
  for (PetscInt j = j_0; j < k; j++) {
    PetscInt Y_idx   = recycle_index(m, j);
    PetscInt StY_idx = ldfp->strategy == MAT_LMVM_DENSE_INPLACE ? Y_idx : history_index(m, k, j);
    Vec      y_j, Sty_j;

    PetscCall(MatDenseGetColumnVecRead(Yfull, Y_idx, &y_j));
    PetscCall(MatDenseGetColumnVecWrite(ldfp->StY_triu_strict, StY_idx, &Sty_j));
    PetscCall(MatMultHermitianTransposeColumnRange(Sfull, y_j, Sty_j, 0, h));
    ldfp->St_count++;
    if (ldfp->strategy == MAT_LMVM_DENSE_REORDER) PetscCall(VecRecycleOrderToHistoryOrder(B, Sty_j, ldfp->num_updates, ldfp->cyclic_work_vec));
    PetscCall(MatDenseRestoreColumnVecWrite(ldfp->StY_triu_strict, StY_idx, &Sty_j));
    PetscCall(MatDenseRestoreColumnVecRead(Yfull, Y_idx, &y_j));
    /* zero the corresponding row */
    if (m_local > 0) {
      Mat StY_local, StY_row;

      PetscCall(MatDenseGetLocalMatrix(ldfp->StY_triu_strict, &StY_local));
      PetscCall(MatDenseGetSubMatrix(StY_local, StY_idx, StY_idx + 1, PETSC_DECIDE, PETSC_DECIDE, &StY_row));
      PetscCall(MatZeroEntries(StY_row));
      PetscCall(MatDenseRestoreSubMatrix(StY_local, &StY_row));
    }
  }
  if (!ldfp->inv_diag_vec) PetscCall(VecDuplicate(ldfp->diag_vec, &ldfp->inv_diag_vec));
  PetscCall(VecCopy(ldfp->diag_vec, ldfp->inv_diag_vec));
  PetscCall(VecReciprocal(ldfp->inv_diag_vec));
  PetscCall(MatDenseGetLocalMatrix(ldfp->J, &J_local));
  PetscCall(MatSetFactorType(J_local, MAT_FACTOR_NONE));
  PetscCall(MatGetRTDR(B, ldfp->J));
  PetscCall(MatAXPY(ldfp->J, 1.0, ldfp->YtHY, SAME_NONZERO_PATTERN));
  if (m_local) {
    PetscCall(MatSetOption(J_local, MAT_SPD, PETSC_TRUE));
    PetscCall(MatCholeskyFactor(J_local, NULL, NULL));
  }
  ldfp->num_mult_updates = ldfp->num_updates;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Solves for

   H_0 - [ S | H_0 Y] [ -D  |    R.T    ]^-1 [   S^T   ]
                      [-----+-----------]    [---------]
                      [  R  | Y^T H_0 Y ]    [ Y^T H_0 ]

   Above is equivalent to

   H_0 - [ S | H_0 Y] [[     I     | 0 ][ -D | 0 ][ I | -D^{-1} R^T ]]^-1 [   S^T   ]
                      [[-----------+---][----+---][---+-------------]]    [---------]
                      [[ -R D^{-1} | I ][  0 | J ][ 0 |      I      ]]    [ Y^T H_0 ]

   where J = Y^T H_0 Y + R D^{-1} R.T

   becomes

   H_0 - [ S | H_0 Y] [ I | D^{-1} R^T ][ -D^{-1}  |   0    ][     I    | 0 ] [   S^T   ]
                      [---+------------][----------+--------][----------+---] [---------]
                      [ 0 |      I     ][     0    | J^{-1} ][ R D^{-1} | I ] [ Y^T H_0 ]

                      =

   H_0 + [ S | H_0 Y] [ D^{-1} | 0 ][ I | R^T ][ I |    0    ][     I    | 0 ] [   S^T   ]
                      [--------+---][---+-----][---+---------][----------+---] [---------]
                      [ 0      | I ][ 0 |  I  ][ 0 | -J^{-1} ][ R D^{-1} | I ] [ Y^T H_0 ]

                      (Note that StY_triu_strict is R)
   Byrd, Nocedal, Schnabel 1994

*/
static PetscErrorCode MatSolve_LMVMDDFP(Mat H, Vec F, Vec dX)
{
  Mat_LMVM *lmvm = (Mat_LMVM *)H->data;
  Mat_DQN  *ldfp = (Mat_DQN *)lmvm->ctx;
  PetscInt  m    = lmvm->m;
  PetscInt  k    = lmvm->k;
  PetscInt  h    = k - oldest_update(m, k);
  PetscInt  idx, i, j, local_n;
  PetscInt  m_local;
  Mat       J_local;
  Mat       Sfull = lmvm->basis[LMBASIS_S]->vecs;
  Mat       Yfull = lmvm->basis[LMBASIS_Y]->vecs;

  PetscFunctionBegin;
  VecCheckSameSize(F, 2, dX, 3);
  VecCheckMatCompatible(H, dX, 3, F, 2);

  /* Cholesky Version */
  /* Start with the B0 term */
  PetscCall(MatDQNApplyJ0Inv(H, F, dX));
  if (!ldfp->num_updates) PetscFunctionReturn(PETSC_SUCCESS); /* No updates stored yet */

  if (ldfp->use_recursive) {
    PetscDeviceContext dctx;
    PetscMemType       memtype;
    PetscScalar        stf, ytx, ytq, yjtqi, sjtyi, *workscalar;

    PetscCall(PetscDeviceContextGetCurrentContext(&dctx));
    /* Recursive formulation to avoid Cholesky. Not a dense formulation */
    PetscCall(MatMultHermitianTransposeColumnRange(Sfull, F, ldfp->rwork1, 0, h));
    ldfp->Yt_count++;

    PetscCall(VecGetLocalSize(ldfp->rwork1, &local_n));

    PetscInt oldest = oldest_update(m, k);

    if (ldfp->needPQ) {
      PetscInt oldest = oldest_update(m, k);
      for (i = oldest; i < k; ++i) {
        idx = recycle_index(m, i);
        /* column_work = S[idx] */
        PetscCall(MatGetColumnVector(Yfull, ldfp->column_work, idx));
        PetscCall(MatDQNApplyJ0Inv(H, ldfp->column_work, ldfp->PQ[idx]));
        PetscCall(MatMultHermitianTransposeColumnRange(Sfull, ldfp->column_work, ldfp->rwork3, 0, h));
        PetscCall(VecGetArrayAndMemType(ldfp->rwork3, &workscalar, &memtype));
        for (j = oldest; j < i; ++j) {
          PetscInt idx_j = recycle_index(m, j);
          /* Copy sjtyi in device-aware manner */
          if (local_n) {
            if (PetscMemTypeHost(memtype)) {
              sjtyi = workscalar[idx_j];
            } else {
              PetscCall(PetscDeviceRegisterMemory(&sjtyi, PETSC_MEMTYPE_HOST, 1 * sizeof(sjtyi)));
              PetscCall(PetscDeviceRegisterMemory(workscalar, memtype, local_n * sizeof(*workscalar)));
              PetscCall(PetscDeviceArrayCopy(dctx, &sjtyi, &workscalar[idx_j], 1));
            }
          }
          PetscCallMPI(MPI_Bcast(&sjtyi, 1, MPIU_SCALAR, 0, PetscObjectComm((PetscObject)H)));
          /* column_work2 = Y[j] */
          PetscCall(MatGetColumnVector(Yfull, ldfp->column_work2, idx_j));
          PetscCall(VecDot(ldfp->PQ[idx], ldfp->column_work2, &yjtqi));
          /* column_work2 = Y[j] */
          PetscCall(MatGetColumnVector(Sfull, ldfp->column_work2, idx_j));
          /* Compute the pure BFGS component of the forward product */
          PetscCall(VecAXPBYPCZ(ldfp->PQ[idx], -yjtqi / ldfp->ytq[idx_j], sjtyi / ldfp->yts[idx_j], 1.0, ldfp->PQ[idx_j], ldfp->column_work2));
        }
        PetscCall(VecDot(ldfp->PQ[idx], ldfp->column_work, &ytq));
        ldfp->ytq[idx] = PetscRealPart(ytq);
      }
      ldfp->needPQ = PETSC_FALSE;
    }

    PetscCall(VecGetArrayAndMemType(ldfp->rwork1, &workscalar, &memtype));
    for (i = oldest; i < k; ++i) {
      idx = recycle_index(m, i);
      /* Copy stz[i], ytx[i] in device-aware manner */
      if (local_n) {
        if (PetscMemTypeHost(memtype)) {
          stf = workscalar[idx];
        } else {
          PetscCall(PetscDeviceRegisterMemory(&stf, PETSC_MEMTYPE_HOST, sizeof(stf)));
          PetscCall(PetscDeviceRegisterMemory(workscalar, memtype, local_n * sizeof(*workscalar)));
          PetscCall(PetscDeviceArrayCopy(dctx, &stf, &workscalar[idx], 1));
        }
      }
      PetscCallMPI(MPI_Bcast(&stf, 1, MPIU_SCALAR, 0, PetscObjectComm((PetscObject)H)));
      /* column_work : S[i], column_work2 : Y[i] */
      PetscCall(MatGetColumnVector(Sfull, ldfp->column_work, idx));
      PetscCall(MatGetColumnVector(Yfull, ldfp->column_work2, idx));
      PetscCall(VecDot(dX, ldfp->column_work2, &ytx));
      PetscCall(VecAXPBYPCZ(dX, -ytx / ldfp->ytq[idx], stf / ldfp->yts[idx], 1.0, ldfp->PQ[idx], ldfp->column_work));
    }
    PetscCall(VecRestoreArrayAndMemType(ldfp->rwork1, &workscalar));
  } else {
    PetscCall(MatLMVMDDFPUpdateSolveData(H));
    PetscCall(MatMultHermitianTransposeColumnRange(Sfull, F, ldfp->rwork1, 0, h));
    ldfp->St_count++;
    PetscCall(MatMultHermitianTransposeColumnRange(Yfull, dX, ldfp->rwork2, 0, h));
    ldfp->Yt_count++;
    if (ldfp->strategy == MAT_LMVM_DENSE_REORDER) {
      PetscCall(VecRecycleOrderToHistoryOrder(H, ldfp->rwork1, ldfp->num_updates, ldfp->cyclic_work_vec));
      PetscCall(VecRecycleOrderToHistoryOrder(H, ldfp->rwork2, ldfp->num_updates, ldfp->cyclic_work_vec));
    }

    PetscCall(VecPointwiseMult(ldfp->rwork3, ldfp->rwork1, ldfp->inv_diag_vec));
    if (PetscDefined(USE_COMPLEX)) PetscCall(MatConjugate(ldfp->StY_triu_strict));
    PetscCall(MatMultTransposeAdd(ldfp->StY_triu_strict, ldfp->rwork3, ldfp->rwork2, ldfp->rwork2));
    if (PetscDefined(USE_COMPLEX)) PetscCall(MatConjugate(ldfp->StY_triu_strict));

    if (!ldfp->rwork2_local) PetscCall(VecCreateLocalVector(ldfp->rwork2, &ldfp->rwork2_local));
    if (!ldfp->rwork3_local) PetscCall(VecCreateLocalVector(ldfp->rwork3, &ldfp->rwork3_local));
    PetscCall(VecGetLocalVectorRead(ldfp->rwork2, ldfp->rwork2_local));
    PetscCall(VecGetLocalVector(ldfp->rwork3, ldfp->rwork3_local));
    PetscCall(MatDenseGetLocalMatrix(ldfp->J, &J_local));
    PetscCall(VecGetSize(ldfp->rwork2_local, &m_local));
    if (m_local) {
      Mat J_local;

      PetscCall(MatDenseGetLocalMatrix(ldfp->J, &J_local));
      PetscCall(MatSolve(J_local, ldfp->rwork2_local, ldfp->rwork3_local));
    }
    PetscCall(VecRestoreLocalVector(ldfp->rwork3, ldfp->rwork3_local));
    PetscCall(VecRestoreLocalVectorRead(ldfp->rwork2, ldfp->rwork2_local));
    PetscCall(VecScale(ldfp->rwork3, -1.0));

    PetscCall(MatMultAdd(ldfp->StY_triu_strict, ldfp->rwork3, ldfp->rwork1, ldfp->rwork1));
    PetscCall(VecPointwiseMult(ldfp->rwork1, ldfp->rwork1, ldfp->inv_diag_vec));

    if (ldfp->strategy == MAT_LMVM_DENSE_REORDER) {
      PetscCall(VecHistoryOrderToRecycleOrder(H, ldfp->rwork1, ldfp->num_updates, ldfp->cyclic_work_vec));
      PetscCall(VecHistoryOrderToRecycleOrder(H, ldfp->rwork3, ldfp->num_updates, ldfp->cyclic_work_vec));
    }

    PetscCall(MatMultAddColumnRange(Sfull, ldfp->rwork1, dX, dX, 0, h));
    ldfp->S_count++;
    PetscCall(MatMultAddColumnRange(ldfp->HY, ldfp->rwork3, dX, dX, 0, h));
    ldfp->Y_count++;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Solves for
   (Theorem 1, Erway, Jain, and Marcia, 2013)

   B_0 - [ Y | B_0 S] [ -R^{-T} (D + S^T B_0 S) R^{-1} | R^{-T} ] [   Y^T   ]
                      ---------------------------------+--------] [---------]
                      [             R^{-1}             |   0    ] [ S^T B_0 ]

   (Note: R above is right triangular part of YTS)
   which becomes,

   [ I | -Y L^{-T} ] [  I  | 0 ] [ B_0 | 0 ] [ I | S ] [      I      ]
                     [-----+---] [-----+---] [---+---] [-------------]
                     [ S^T | I ] [  0  | D ] [ 0 | I ] [ -L^{-1} Y^T ]

   (Note: L above is right triangular part of STY)

*/
static PetscErrorCode MatMult_LMVMDDFP(Mat B, Vec X, Vec Z)
{
  Mat_LMVM        *lmvm   = (Mat_LMVM *)B->data;
  Mat_DQN         *ldfp   = (Mat_DQN *)lmvm->ctx;
  Vec              rwork1 = ldfp->rwork1;
  PetscInt         m      = lmvm->m;
  PetscInt         k      = lmvm->k;
  PetscInt         h      = k - oldest_update(m, k);
  Mat              Sfull  = lmvm->basis[LMBASIS_S]->vecs;
  Mat              Yfull  = lmvm->basis[LMBASIS_Y]->vecs;
  PetscObjectState Xstate;

  PetscFunctionBegin;
  VecCheckSameSize(X, 2, Z, 3);
  VecCheckMatCompatible(B, X, 2, Z, 3);

  /* DFP Version. Erway, Jain, Marcia, 2013, Theorem 1 */
  /* Block Version */
  if (!ldfp->num_updates) {
    PetscCall(MatDQNApplyJ0Fwd(B, X, Z));
    PetscFunctionReturn(PETSC_SUCCESS); /* No updates stored yet */
  }

  PetscCall(PetscObjectStateGet((PetscObject)X, &Xstate));
  PetscCall(MatMultHermitianTransposeColumnRange(Yfull, X, rwork1, 0, h));

  /* Reordering rwork1, as STY is in history order, while Y is in recycled order */
  if (ldfp->strategy == MAT_LMVM_DENSE_REORDER) PetscCall(VecRecycleOrderToHistoryOrder(B, rwork1, ldfp->num_updates, ldfp->cyclic_work_vec));
  PetscCall(MatUpperTriangularSolveInPlace(B, ldfp->YtS_triu, rwork1, PETSC_FALSE, ldfp->num_updates, ldfp->strategy));
  PetscCall(VecScale(rwork1, -1.0));
  if (ldfp->strategy == MAT_LMVM_DENSE_REORDER) PetscCall(VecHistoryOrderToRecycleOrder(B, rwork1, ldfp->num_updates, ldfp->cyclic_work_vec));

  PetscCall(VecCopy(X, ldfp->column_work));
  PetscCall(MatMultAddColumnRange(Sfull, rwork1, ldfp->column_work, ldfp->column_work, 0, h));
  ldfp->S_count++;

  PetscCall(VecPointwiseMult(rwork1, ldfp->diag_vec_recycle_order, rwork1));
  PetscCall(MatDQNApplyJ0Fwd(B, ldfp->column_work, Z));

  PetscCall(MatMultHermitianTransposeAddColumnRange(Sfull, Z, rwork1, rwork1, 0, h));
  ldfp->St_count++;

  if (ldfp->strategy == MAT_LMVM_DENSE_REORDER) PetscCall(VecRecycleOrderToHistoryOrder(B, rwork1, ldfp->num_updates, ldfp->cyclic_work_vec));
  PetscCall(MatUpperTriangularSolveInPlace(B, ldfp->YtS_triu, rwork1, PETSC_TRUE, ldfp->num_updates, ldfp->strategy));
  PetscCall(VecScale(rwork1, -1.0));
  if (ldfp->strategy == MAT_LMVM_DENSE_REORDER) PetscCall(VecHistoryOrderToRecycleOrder(B, rwork1, ldfp->num_updates, ldfp->cyclic_work_vec));

  PetscCall(MatMultAddColumnRange(Yfull, rwork1, Z, Z, 0, h));
  ldfp->Y_count++;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
   This dense representation reduces the L-DFP update to a series of
   matrix-vector products with dense matrices in lieu of the conventional
   matrix-free two-loop algorithm.
*/
PetscErrorCode MatCreate_LMVMDDFP(Mat B)
{
  Mat_LMVM *lmvm;
  Mat_DQN  *ldfp;

  PetscFunctionBegin;
  PetscCall(MatCreate_LMVM(B));
  PetscCall(PetscObjectChangeTypeName((PetscObject)B, MATLMVMDDFP));
  PetscCall(MatSetOption(B, MAT_HERMITIAN, PETSC_TRUE));
  PetscCall(MatSetOption(B, MAT_SPD, PETSC_TRUE));
  PetscCall(MatSetOption(B, MAT_SPD_ETERNAL, PETSC_TRUE));
  B->ops->view           = MatView_LMVMDQN;
  B->ops->setup          = MatSetUp_LMVMDQN;
  B->ops->setfromoptions = MatSetFromOptions_LMVMDQN;
  B->ops->destroy        = MatDestroy_LMVMDQN;

  lmvm              = (Mat_LMVM *)B->data;
  lmvm->ops->reset  = MatReset_LMVMDQN;
  lmvm->ops->update = MatUpdate_LMVMDQN;
  lmvm->ops->mult   = MatMult_LMVMDDFP;
  lmvm->ops->solve  = MatSolve_LMVMDDFP;
  lmvm->ops->copy   = MatCopy_LMVMDQN;

  lmvm->ops->multht  = lmvm->ops->mult;
  lmvm->ops->solveht = lmvm->ops->solve;

  PetscCall(PetscNew(&ldfp));
  lmvm->ctx             = (void *)ldfp;
  ldfp->allocated       = PETSC_FALSE;
  ldfp->watchdog        = 0;
  ldfp->max_seq_rejects = lmvm->m / 2;
  ldfp->strategy        = MAT_LMVM_DENSE_INPLACE;
  ldfp->use_recursive   = PETSC_TRUE;
  ldfp->needPQ          = PETSC_TRUE;

  PetscCall(SymBroydenRescaleCreate(&ldfp->rescale));
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatLMVMSymBroydenSetDelta_C", MatLMVMSymBroydenSetDelta_LMVMDQN));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  MatCreateLMVMDDFP - Creates a dense representation of the limited-memory
  Davidon-Fletcher-Powell (DFP) approximation to a Hessian.

  Collective

  Input Parameters:
+ comm - MPI communicator
. n    - number of local rows for storage vectors
- N    - global size of the storage vectors

  Output Parameter:
. B - the matrix

  Level: advanced

  Note:
  It is recommended that one use the MatCreate(), MatSetType() and/or MatSetFromOptions()
  paradigm instead of this routine directly.

.seealso: `MatCreate()`, `MATLMVM`, `MATLMVMDDFP`, `MatCreateLMVMDFP()`
@*/
PetscErrorCode MatCreateLMVMDDFP(MPI_Comm comm, PetscInt n, PetscInt N, Mat *B)
{
  PetscFunctionBegin;
  PetscCall(KSPInitializePackage());
  PetscCall(MatCreate(comm, B));
  PetscCall(MatSetSizes(*B, n, n, N, N));
  PetscCall(MatSetType(*B, MATLMVMDDFP));
  PetscCall(MatSetUp(*B));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  MatLMVMDenseSetType - Sets the memory storage type for dense `MATLMVM`

  Input Parameters:
+ B    - the `MATLMVM` matrix
- type - scale type, see `MatLMVMDenseSetType`

  Options Database Keys:
+ -mat_lqn_type   <reorder,inplace> - set the strategy
. -mat_lbfgs_type <reorder,inplace> - set the strategy
- -mat_ldfp_type  <reorder,inplace> - set the strategy

  Level: intermediate

  MatLMVMDenseTypes\:
+   `MAT_LMVM_DENSE_REORDER` - reorders memory to minimize kernel launch
-   `MAT_LMVM_DENSE_INPLACE` - launches kernel inplace to minimize memory movement

.seealso: [](ch_ksp), `MATLMVMDQN`, `MATLMVMDBFGS`, `MATLMVMDDFP`, `MatLMVMDenseType`
@*/
PetscErrorCode MatLMVMDenseSetType(Mat B, MatLMVMDenseType type)
{
  Mat_LMVM *lmvm = (Mat_LMVM *)B->data;
  Mat_DQN  *lqn  = (Mat_DQN *)lmvm->ctx;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(B, MAT_CLASSID, 1);
  lqn->strategy = type;
  PetscFunctionReturn(PETSC_SUCCESS);
}

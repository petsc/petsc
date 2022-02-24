#include <../src/tao/bound/impls/bqnk/bqnk.h> /*I "petsctao.h" I*/ /*I "petscmat.h" I*/
#include <petscksp.h>

static PetscErrorCode TaoBQNKComputeHessian(Tao tao)
{
  TAO_BNK        *bnk = (TAO_BNK *)tao->data;
  TAO_BQNK       *bqnk = (TAO_BQNK*)bnk->ctx;
  PetscReal      gnorm2, delta;

  PetscFunctionBegin;
  /* Alias the LMVM matrix into the TAO hessian */
  if (tao->hessian) {
    CHKERRQ(MatDestroy(&tao->hessian));
  }
  if (tao->hessian_pre) {
    CHKERRQ(MatDestroy(&tao->hessian_pre));
  }
  CHKERRQ(PetscObjectReference((PetscObject)bqnk->B));
  tao->hessian = bqnk->B;
  CHKERRQ(PetscObjectReference((PetscObject)bqnk->B));
  tao->hessian_pre = bqnk->B;
  /* Update the Hessian with the latest solution */
  if (bqnk->is_spd) {
    gnorm2 = bnk->gnorm*bnk->gnorm;
    if (gnorm2 == 0.0) gnorm2 = PETSC_MACHINE_EPSILON;
    if (bnk->f == 0.0) {
      delta = 2.0 / gnorm2;
    } else {
      delta = 2.0 * PetscAbsScalar(bnk->f) / gnorm2;
    }
    CHKERRQ(MatLMVMSymBroydenSetDelta(bqnk->B, delta));
  }
  CHKERRQ(MatLMVMUpdate(tao->hessian, tao->solution, bnk->unprojected_gradient));
  CHKERRQ(MatLMVMResetShift(tao->hessian));
  /* Prepare the reduced sub-matrices for the inactive set */
  CHKERRQ(MatDestroy(&bnk->H_inactive));
  if (bnk->active_idx) {
    CHKERRQ(MatCreateSubMatrixVirtual(tao->hessian, bnk->inactive_idx, bnk->inactive_idx, &bnk->H_inactive));
    CHKERRQ(PCLMVMSetIS(bqnk->pc, bnk->inactive_idx));
  } else {
    CHKERRQ(PetscObjectReference((PetscObject)tao->hessian));
    bnk->H_inactive = tao->hessian;
    CHKERRQ(PCLMVMClearIS(bqnk->pc));
  }
  CHKERRQ(MatDestroy(&bnk->Hpre_inactive));
  CHKERRQ(PetscObjectReference((PetscObject)bnk->H_inactive));
  bnk->Hpre_inactive = bnk->H_inactive;
  PetscFunctionReturn(0);
}

static PetscErrorCode TaoBQNKComputeStep(Tao tao, PetscBool shift, KSPConvergedReason *ksp_reason, PetscInt *step_type)
{
  TAO_BNK        *bnk = (TAO_BNK *)tao->data;
  TAO_BQNK       *bqnk = (TAO_BQNK*)bnk->ctx;

  PetscFunctionBegin;
  CHKERRQ(TaoBNKComputeStep(tao, shift, ksp_reason, step_type));
  if (*ksp_reason < 0) {
    /* Krylov solver failed to converge so reset the LMVM matrix */
    CHKERRQ(MatLMVMReset(bqnk->B, PETSC_FALSE));
    CHKERRQ(MatLMVMUpdate(bqnk->B, tao->solution, bnk->unprojected_gradient));
  }
  PetscFunctionReturn(0);
}

PetscErrorCode TaoSolve_BQNK(Tao tao)
{
  TAO_BNK        *bnk = (TAO_BNK *)tao->data;
  TAO_BQNK       *bqnk = (TAO_BQNK*)bnk->ctx;
  Mat_LMVM       *lmvm = (Mat_LMVM*)bqnk->B->data;
  Mat_LMVM       *J0;
  Mat_SymBrdn    *diag_ctx;
  PetscBool      flg = PETSC_FALSE;

  PetscFunctionBegin;
  if (!tao->recycle) {
    CHKERRQ(MatLMVMReset(bqnk->B, PETSC_FALSE));
    lmvm->nresets = 0;
    if (lmvm->J0) {
      CHKERRQ(PetscObjectBaseTypeCompare((PetscObject)lmvm->J0, MATLMVM, &flg));
      if (flg) {
        J0 = (Mat_LMVM*)lmvm->J0->data;
        J0->nresets = 0;
      }
    }
    flg = PETSC_FALSE;
    CHKERRQ(PetscObjectTypeCompareAny((PetscObject)bqnk->B, &flg, MATLMVMSYMBROYDEN, MATLMVMSYMBADBROYDEN, MATLMVMBFGS, MATLMVMDFP, ""));
    if (flg) {
      diag_ctx = (Mat_SymBrdn*)lmvm->ctx;
      J0 = (Mat_LMVM*)diag_ctx->D->data;
      J0->nresets = 0;
    }
  }
  CHKERRQ((*bqnk->solve)(tao));
  PetscFunctionReturn(0);
}

PetscErrorCode TaoSetUp_BQNK(Tao tao)
{
  TAO_BNK        *bnk = (TAO_BNK *)tao->data;
  TAO_BQNK       *bqnk = (TAO_BQNK*)bnk->ctx;
  PetscInt       n, N;
  PetscBool      is_lmvm, is_sym, is_spd;

  PetscFunctionBegin;
  CHKERRQ(TaoSetUp_BNK(tao));
  CHKERRQ(VecGetLocalSize(tao->solution,&n));
  CHKERRQ(VecGetSize(tao->solution,&N));
  CHKERRQ(MatSetSizes(bqnk->B, n, n, N, N));
  CHKERRQ(MatLMVMAllocate(bqnk->B,tao->solution,bnk->unprojected_gradient));
  CHKERRQ(PetscObjectBaseTypeCompare((PetscObject)bqnk->B, MATLMVM, &is_lmvm));
  PetscCheck(is_lmvm,PetscObjectComm((PetscObject)tao), PETSC_ERR_ARG_INCOMP, "Matrix must be an LMVM-type");
  CHKERRQ(MatGetOption(bqnk->B, MAT_SYMMETRIC, &is_sym));
  PetscCheck(is_sym,PetscObjectComm((PetscObject)tao), PETSC_ERR_ARG_INCOMP, "LMVM matrix must be symmetric");
  CHKERRQ(MatGetOption(bqnk->B, MAT_SPD, &is_spd));
  CHKERRQ(KSPGetPC(tao->ksp, &bqnk->pc));
  CHKERRQ(PCSetType(bqnk->pc, PCLMVM));
  CHKERRQ(PCLMVMSetMatLMVM(bqnk->pc, bqnk->B));
  PetscFunctionReturn(0);
}

static PetscErrorCode TaoSetFromOptions_BQNK(PetscOptionItems *PetscOptionsObject,Tao tao)
{
  TAO_BNK        *bnk = (TAO_BNK *)tao->data;
  TAO_BQNK       *bqnk = (TAO_BQNK*)bnk->ctx;

  PetscFunctionBegin;
  CHKERRQ(TaoSetFromOptions_BNK(PetscOptionsObject,tao));
  if (bnk->init_type == BNK_INIT_INTERPOLATION) bnk->init_type = BNK_INIT_DIRECTION;
  CHKERRQ(MatSetOptionsPrefix(bqnk->B, ((PetscObject)tao)->prefix));
  CHKERRQ(MatAppendOptionsPrefix(bqnk->B, "tao_bqnk_"));
  CHKERRQ(MatSetFromOptions(bqnk->B));
  CHKERRQ(MatGetOption(bqnk->B, MAT_SPD, &bqnk->is_spd));
  PetscFunctionReturn(0);
}

static PetscErrorCode TaoView_BQNK(Tao tao, PetscViewer viewer)
{
  TAO_BNK        *bnk = (TAO_BNK*)tao->data;
  TAO_BQNK       *bqnk = (TAO_BQNK*)bnk->ctx;
  PetscBool      isascii;

  PetscFunctionBegin;
  CHKERRQ(TaoView_BNK(tao, viewer));
  CHKERRQ(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&isascii));
  if (isascii) {
    CHKERRQ(PetscViewerPushFormat(viewer, PETSC_VIEWER_ASCII_INFO));
    CHKERRQ(MatView(bqnk->B, viewer));
    CHKERRQ(PetscViewerPopFormat(viewer));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode TaoDestroy_BQNK(Tao tao)
{
  TAO_BNK        *bnk = (TAO_BNK*)tao->data;
  TAO_BQNK       *bqnk = (TAO_BQNK*)bnk->ctx;

  PetscFunctionBegin;
  CHKERRQ(MatDestroy(&bnk->Hpre_inactive));
  CHKERRQ(MatDestroy(&bnk->H_inactive));
  CHKERRQ(MatDestroy(&bqnk->B));
  CHKERRQ(PetscFree(bnk->ctx));
  CHKERRQ(TaoDestroy_BNK(tao));
  PetscFunctionReturn(0);
}

PETSC_INTERN PetscErrorCode TaoCreate_BQNK(Tao tao)
{
  TAO_BNK        *bnk;
  TAO_BQNK       *bqnk;

  PetscFunctionBegin;
  CHKERRQ(TaoCreate_BNK(tao));
  tao->ops->solve = TaoSolve_BQNK;
  tao->ops->setfromoptions = TaoSetFromOptions_BQNK;
  tao->ops->destroy = TaoDestroy_BQNK;
  tao->ops->view = TaoView_BQNK;
  tao->ops->setup = TaoSetUp_BQNK;

  bnk = (TAO_BNK *)tao->data;
  bnk->computehessian = TaoBQNKComputeHessian;
  bnk->computestep = TaoBQNKComputeStep;
  bnk->init_type = BNK_INIT_DIRECTION;

  CHKERRQ(PetscNewLog(tao,&bqnk));
  bnk->ctx = (void*)bqnk;
  bqnk->is_spd = PETSC_TRUE;

  CHKERRQ(MatCreate(PetscObjectComm((PetscObject)tao), &bqnk->B));
  CHKERRQ(PetscObjectIncrementTabLevel((PetscObject)bqnk->B, (PetscObject)tao, 1));
  CHKERRQ(MatSetType(bqnk->B, MATLMVMSR1));
  PetscFunctionReturn(0);
}

/*@
   TaoGetLMVMMatrix - Returns a pointer to the internal LMVM matrix. Valid
   only for quasi-Newton family of methods.

   Input Parameters:
.  tao - Tao solver context

   Output Parameters:
.  B - LMVM matrix

   Level: advanced

.seealso: TAOBQNLS, TAOBQNKLS, TAOBQNKTL, TAOBQNKTR, MATLMVM, TaoSetLMVMMatrix()
@*/
PetscErrorCode TaoGetLMVMMatrix(Tao tao, Mat *B)
{
  TAO_BNK        *bnk = (TAO_BNK*)tao->data;
  TAO_BQNK       *bqnk = (TAO_BQNK*)bnk->ctx;
  PetscBool      flg = PETSC_FALSE;

  PetscFunctionBegin;
  CHKERRQ(PetscObjectTypeCompareAny((PetscObject)tao, &flg, TAOBQNLS, TAOBQNKLS, TAOBQNKTR, TAOBQNKTL, ""));
  PetscCheck(flg,PetscObjectComm((PetscObject)tao), PETSC_ERR_ARG_INCOMP, "LMVM Matrix only exists for quasi-Newton algorithms");
  *B = bqnk->B;
  PetscFunctionReturn(0);
}

/*@
   TaoSetLMVMMatrix - Sets an external LMVM matrix into the Tao solver. Valid
   only for quasi-Newton family of methods.

   QN family of methods create their own LMVM matrices and users who wish to
   manipulate this matrix should use TaoGetLMVMMatrix() instead.

   Input Parameters:
+  tao - Tao solver context
-  B - LMVM matrix

   Level: advanced

.seealso: TAOBQNLS, TAOBQNKLS, TAOBQNKTL, TAOBQNKTR, MATLMVM, TaoGetLMVMMatrix()
@*/
PetscErrorCode TaoSetLMVMMatrix(Tao tao, Mat B)
{
  TAO_BNK        *bnk = (TAO_BNK*)tao->data;
  TAO_BQNK       *bqnk = (TAO_BQNK*)bnk->ctx;
  PetscBool      flg = PETSC_FALSE;

  PetscFunctionBegin;
  CHKERRQ(PetscObjectTypeCompareAny((PetscObject)tao, &flg, TAOBQNLS, TAOBQNKLS, TAOBQNKTR, TAOBQNKTL, ""));
  PetscCheck(flg,PetscObjectComm((PetscObject)tao), PETSC_ERR_ARG_INCOMP, "LMVM Matrix only exists for quasi-Newton algorithms");
  CHKERRQ(PetscObjectBaseTypeCompare((PetscObject)B, MATLMVM, &flg));
  PetscCheck(flg,PetscObjectComm((PetscObject)tao), PETSC_ERR_ARG_INCOMP, "Given matrix is not an LMVM matrix");
  if (bqnk->B) {
    CHKERRQ(MatDestroy(&bqnk->B));
  }
  CHKERRQ(PetscObjectReference((PetscObject)B));
  bqnk->B = B;
  PetscFunctionReturn(0);
}

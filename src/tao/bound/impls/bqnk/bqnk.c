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
    PetscCall(MatDestroy(&tao->hessian));
  }
  if (tao->hessian_pre) {
    PetscCall(MatDestroy(&tao->hessian_pre));
  }
  PetscCall(PetscObjectReference((PetscObject)bqnk->B));
  tao->hessian = bqnk->B;
  PetscCall(PetscObjectReference((PetscObject)bqnk->B));
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
    PetscCall(MatLMVMSymBroydenSetDelta(bqnk->B, delta));
  }
  PetscCall(MatLMVMUpdate(tao->hessian, tao->solution, bnk->unprojected_gradient));
  PetscCall(MatLMVMResetShift(tao->hessian));
  /* Prepare the reduced sub-matrices for the inactive set */
  PetscCall(MatDestroy(&bnk->H_inactive));
  if (bnk->active_idx) {
    PetscCall(MatCreateSubMatrixVirtual(tao->hessian, bnk->inactive_idx, bnk->inactive_idx, &bnk->H_inactive));
    PetscCall(PCLMVMSetIS(bqnk->pc, bnk->inactive_idx));
  } else {
    PetscCall(PetscObjectReference((PetscObject)tao->hessian));
    bnk->H_inactive = tao->hessian;
    PetscCall(PCLMVMClearIS(bqnk->pc));
  }
  PetscCall(MatDestroy(&bnk->Hpre_inactive));
  PetscCall(PetscObjectReference((PetscObject)bnk->H_inactive));
  bnk->Hpre_inactive = bnk->H_inactive;
  PetscFunctionReturn(0);
}

static PetscErrorCode TaoBQNKComputeStep(Tao tao, PetscBool shift, KSPConvergedReason *ksp_reason, PetscInt *step_type)
{
  TAO_BNK        *bnk = (TAO_BNK *)tao->data;
  TAO_BQNK       *bqnk = (TAO_BQNK*)bnk->ctx;

  PetscFunctionBegin;
  PetscCall(TaoBNKComputeStep(tao, shift, ksp_reason, step_type));
  if (*ksp_reason < 0) {
    /* Krylov solver failed to converge so reset the LMVM matrix */
    PetscCall(MatLMVMReset(bqnk->B, PETSC_FALSE));
    PetscCall(MatLMVMUpdate(bqnk->B, tao->solution, bnk->unprojected_gradient));
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
    PetscCall(MatLMVMReset(bqnk->B, PETSC_FALSE));
    lmvm->nresets = 0;
    if (lmvm->J0) {
      PetscCall(PetscObjectBaseTypeCompare((PetscObject)lmvm->J0, MATLMVM, &flg));
      if (flg) {
        J0 = (Mat_LMVM*)lmvm->J0->data;
        J0->nresets = 0;
      }
    }
    flg = PETSC_FALSE;
    PetscCall(PetscObjectTypeCompareAny((PetscObject)bqnk->B, &flg, MATLMVMSYMBROYDEN, MATLMVMSYMBADBROYDEN, MATLMVMBFGS, MATLMVMDFP, ""));
    if (flg) {
      diag_ctx = (Mat_SymBrdn*)lmvm->ctx;
      J0 = (Mat_LMVM*)diag_ctx->D->data;
      J0->nresets = 0;
    }
  }
  PetscCall((*bqnk->solve)(tao));
  PetscFunctionReturn(0);
}

PetscErrorCode TaoSetUp_BQNK(Tao tao)
{
  TAO_BNK        *bnk = (TAO_BNK *)tao->data;
  TAO_BQNK       *bqnk = (TAO_BQNK*)bnk->ctx;
  PetscInt       n, N;
  PetscBool      is_lmvm, is_sym, is_spd;

  PetscFunctionBegin;
  PetscCall(TaoSetUp_BNK(tao));
  PetscCall(VecGetLocalSize(tao->solution,&n));
  PetscCall(VecGetSize(tao->solution,&N));
  PetscCall(MatSetSizes(bqnk->B, n, n, N, N));
  PetscCall(MatLMVMAllocate(bqnk->B,tao->solution,bnk->unprojected_gradient));
  PetscCall(PetscObjectBaseTypeCompare((PetscObject)bqnk->B, MATLMVM, &is_lmvm));
  PetscCheck(is_lmvm,PetscObjectComm((PetscObject)tao), PETSC_ERR_ARG_INCOMP, "Matrix must be an LMVM-type");
  PetscCall(MatGetOption(bqnk->B, MAT_SYMMETRIC, &is_sym));
  PetscCheck(is_sym,PetscObjectComm((PetscObject)tao), PETSC_ERR_ARG_INCOMP, "LMVM matrix must be symmetric");
  PetscCall(MatGetOption(bqnk->B, MAT_SPD, &is_spd));
  PetscCall(KSPGetPC(tao->ksp, &bqnk->pc));
  PetscCall(PCSetType(bqnk->pc, PCLMVM));
  PetscCall(PCLMVMSetMatLMVM(bqnk->pc, bqnk->B));
  PetscFunctionReturn(0);
}

static PetscErrorCode TaoSetFromOptions_BQNK(PetscOptionItems *PetscOptionsObject,Tao tao)
{
  TAO_BNK        *bnk = (TAO_BNK *)tao->data;
  TAO_BQNK       *bqnk = (TAO_BQNK*)bnk->ctx;

  PetscFunctionBegin;
  PetscCall(TaoSetFromOptions_BNK(PetscOptionsObject,tao));
  if (bnk->init_type == BNK_INIT_INTERPOLATION) bnk->init_type = BNK_INIT_DIRECTION;
  PetscCall(MatSetOptionsPrefix(bqnk->B, ((PetscObject)tao)->prefix));
  PetscCall(MatAppendOptionsPrefix(bqnk->B, "tao_bqnk_"));
  PetscCall(MatSetFromOptions(bqnk->B));
  PetscCall(MatGetOption(bqnk->B, MAT_SPD, &bqnk->is_spd));
  PetscFunctionReturn(0);
}

static PetscErrorCode TaoView_BQNK(Tao tao, PetscViewer viewer)
{
  TAO_BNK        *bnk = (TAO_BNK*)tao->data;
  TAO_BQNK       *bqnk = (TAO_BQNK*)bnk->ctx;
  PetscBool      isascii;

  PetscFunctionBegin;
  PetscCall(TaoView_BNK(tao, viewer));
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&isascii));
  if (isascii) {
    PetscCall(PetscViewerPushFormat(viewer, PETSC_VIEWER_ASCII_INFO));
    PetscCall(MatView(bqnk->B, viewer));
    PetscCall(PetscViewerPopFormat(viewer));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode TaoDestroy_BQNK(Tao tao)
{
  TAO_BNK        *bnk = (TAO_BNK*)tao->data;
  TAO_BQNK       *bqnk = (TAO_BQNK*)bnk->ctx;

  PetscFunctionBegin;
  PetscCall(MatDestroy(&bnk->Hpre_inactive));
  PetscCall(MatDestroy(&bnk->H_inactive));
  PetscCall(MatDestroy(&bqnk->B));
  PetscCall(PetscFree(bnk->ctx));
  PetscCall(TaoDestroy_BNK(tao));
  PetscFunctionReturn(0);
}

PETSC_INTERN PetscErrorCode TaoCreate_BQNK(Tao tao)
{
  TAO_BNK        *bnk;
  TAO_BQNK       *bqnk;

  PetscFunctionBegin;
  PetscCall(TaoCreate_BNK(tao));
  tao->ops->solve = TaoSolve_BQNK;
  tao->ops->setfromoptions = TaoSetFromOptions_BQNK;
  tao->ops->destroy = TaoDestroy_BQNK;
  tao->ops->view = TaoView_BQNK;
  tao->ops->setup = TaoSetUp_BQNK;

  bnk = (TAO_BNK *)tao->data;
  bnk->computehessian = TaoBQNKComputeHessian;
  bnk->computestep = TaoBQNKComputeStep;
  bnk->init_type = BNK_INIT_DIRECTION;

  PetscCall(PetscNewLog(tao,&bqnk));
  bnk->ctx = (void*)bqnk;
  bqnk->is_spd = PETSC_TRUE;

  PetscCall(MatCreate(PetscObjectComm((PetscObject)tao), &bqnk->B));
  PetscCall(PetscObjectIncrementTabLevel((PetscObject)bqnk->B, (PetscObject)tao, 1));
  PetscCall(MatSetType(bqnk->B, MATLMVMSR1));
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
  PetscCall(PetscObjectTypeCompareAny((PetscObject)tao, &flg, TAOBQNLS, TAOBQNKLS, TAOBQNKTR, TAOBQNKTL, ""));
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
  PetscCall(PetscObjectTypeCompareAny((PetscObject)tao, &flg, TAOBQNLS, TAOBQNKLS, TAOBQNKTR, TAOBQNKTL, ""));
  PetscCheck(flg,PetscObjectComm((PetscObject)tao), PETSC_ERR_ARG_INCOMP, "LMVM Matrix only exists for quasi-Newton algorithms");
  PetscCall(PetscObjectBaseTypeCompare((PetscObject)B, MATLMVM, &flg));
  PetscCheck(flg,PetscObjectComm((PetscObject)tao), PETSC_ERR_ARG_INCOMP, "Given matrix is not an LMVM matrix");
  if (bqnk->B) {
    PetscCall(MatDestroy(&bqnk->B));
  }
  PetscCall(PetscObjectReference((PetscObject)B));
  bqnk->B = B;
  PetscFunctionReturn(0);
}

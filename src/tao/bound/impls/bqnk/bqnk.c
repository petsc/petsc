#include <../src/tao/bound/impls/bqnk/bqnk.h> /*I "petsctao.h" I*/ /*I "petscmat.h" I*/
#include <petscksp.h>

static PetscErrorCode TaoBQNKComputeHessian(Tao tao)
{
  TAO_BNK        *bnk = (TAO_BNK *)tao->data;
  TAO_BQNK       *bqnk = (TAO_BQNK*)bnk->ctx;
  PetscErrorCode ierr;
  PetscReal      gnorm2, delta;

  PetscFunctionBegin;
  /* Alias the LMVM matrix into the TAO hessian */
  if (tao->hessian) {
    ierr = MatDestroy(&tao->hessian);CHKERRQ(ierr);
  }
  if (tao->hessian_pre) {
    ierr = MatDestroy(&tao->hessian_pre);CHKERRQ(ierr);
  }
  ierr = PetscObjectReference((PetscObject)bqnk->B);CHKERRQ(ierr);
  tao->hessian = bqnk->B;
  ierr = PetscObjectReference((PetscObject)bqnk->B);CHKERRQ(ierr);
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
    ierr = MatLMVMSymBroydenSetDelta(bqnk->B, delta);CHKERRQ(ierr);
  }
  ierr = MatLMVMUpdate(tao->hessian, tao->solution, bnk->unprojected_gradient);CHKERRQ(ierr);
  ierr = MatLMVMResetShift(tao->hessian);CHKERRQ(ierr);
  /* Prepare the reduced sub-matrices for the inactive set */
  ierr = MatDestroy(&bnk->H_inactive);CHKERRQ(ierr);
  if (bnk->active_idx) {
    ierr = MatCreateSubMatrixVirtual(tao->hessian, bnk->inactive_idx, bnk->inactive_idx, &bnk->H_inactive);CHKERRQ(ierr);
    ierr = PCLMVMSetIS(bqnk->pc, bnk->inactive_idx);CHKERRQ(ierr);
  } else {
    ierr = PetscObjectReference((PetscObject)tao->hessian);CHKERRQ(ierr);
    bnk->H_inactive = tao->hessian;
    ierr = PCLMVMClearIS(bqnk->pc);CHKERRQ(ierr);
  }
  ierr = MatDestroy(&bnk->Hpre_inactive);CHKERRQ(ierr);
  ierr = PetscObjectReference((PetscObject)bnk->H_inactive);CHKERRQ(ierr);
  bnk->Hpre_inactive = bnk->H_inactive;
  PetscFunctionReturn(0);
}

static PetscErrorCode TaoBQNKComputeStep(Tao tao, PetscBool shift, KSPConvergedReason *ksp_reason, PetscInt *step_type)
{
  TAO_BNK        *bnk = (TAO_BNK *)tao->data;
  TAO_BQNK       *bqnk = (TAO_BQNK*)bnk->ctx;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = TaoBNKComputeStep(tao, shift, ksp_reason, step_type);CHKERRQ(ierr);
  if (*ksp_reason < 0) {
    /* Krylov solver failed to converge so reset the LMVM matrix */
    ierr = MatLMVMReset(bqnk->B, PETSC_FALSE);CHKERRQ(ierr);
    ierr = MatLMVMUpdate(bqnk->B, tao->solution, bnk->unprojected_gradient);CHKERRQ(ierr);
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!tao->recycle) {
    ierr = MatLMVMReset(bqnk->B, PETSC_FALSE);CHKERRQ(ierr);
    lmvm->nresets = 0;
    if (lmvm->J0) {
      ierr = PetscObjectBaseTypeCompare((PetscObject)lmvm->J0, MATLMVM, &flg);CHKERRQ(ierr);
      if (flg) {
        J0 = (Mat_LMVM*)lmvm->J0->data;
        J0->nresets = 0;
      }
    }
    flg = PETSC_FALSE;
    ierr = PetscObjectTypeCompareAny((PetscObject)bqnk->B, &flg, MATLMVMSYMBROYDEN, MATLMVMSYMBADBROYDEN, MATLMVMBFGS, MATLMVMDFP, "");CHKERRQ(ierr);
    if (flg) {
      diag_ctx = (Mat_SymBrdn*)lmvm->ctx;
      J0 = (Mat_LMVM*)diag_ctx->D->data;
      J0->nresets = 0;
    }
  }
  ierr = (*bqnk->solve)(tao);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode TaoSetUp_BQNK(Tao tao)
{
  TAO_BNK        *bnk = (TAO_BNK *)tao->data;
  TAO_BQNK       *bqnk = (TAO_BQNK*)bnk->ctx;
  PetscErrorCode ierr;
  PetscInt       n, N;
  PetscBool      is_lmvm, is_sym, is_spd;

  PetscFunctionBegin;
  ierr = TaoSetUp_BNK(tao);CHKERRQ(ierr);
  ierr = VecGetLocalSize(tao->solution,&n);CHKERRQ(ierr);
  ierr = VecGetSize(tao->solution,&N);CHKERRQ(ierr);
  ierr = MatSetSizes(bqnk->B, n, n, N, N);CHKERRQ(ierr);
  ierr = MatLMVMAllocate(bqnk->B,tao->solution,bnk->unprojected_gradient);CHKERRQ(ierr);
  ierr = PetscObjectBaseTypeCompare((PetscObject)bqnk->B, MATLMVM, &is_lmvm);CHKERRQ(ierr);
  PetscCheck(is_lmvm,PetscObjectComm((PetscObject)tao), PETSC_ERR_ARG_INCOMP, "Matrix must be an LMVM-type");
  ierr = MatGetOption(bqnk->B, MAT_SYMMETRIC, &is_sym);CHKERRQ(ierr);
  PetscCheck(is_sym,PetscObjectComm((PetscObject)tao), PETSC_ERR_ARG_INCOMP, "LMVM matrix must be symmetric");
  ierr = MatGetOption(bqnk->B, MAT_SPD, &is_spd);CHKERRQ(ierr);
  ierr = KSPGetPC(tao->ksp, &bqnk->pc);CHKERRQ(ierr);
  ierr = PCSetType(bqnk->pc, PCLMVM);CHKERRQ(ierr);
  ierr = PCLMVMSetMatLMVM(bqnk->pc, bqnk->B);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode TaoSetFromOptions_BQNK(PetscOptionItems *PetscOptionsObject,Tao tao)
{
  TAO_BNK        *bnk = (TAO_BNK *)tao->data;
  TAO_BQNK       *bqnk = (TAO_BQNK*)bnk->ctx;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = TaoSetFromOptions_BNK(PetscOptionsObject,tao);CHKERRQ(ierr);
  if (bnk->init_type == BNK_INIT_INTERPOLATION) bnk->init_type = BNK_INIT_DIRECTION;
  ierr = MatSetOptionsPrefix(bqnk->B, ((PetscObject)tao)->prefix);CHKERRQ(ierr);
  ierr = MatAppendOptionsPrefix(bqnk->B, "tao_bqnk_");CHKERRQ(ierr);
  ierr = MatSetFromOptions(bqnk->B);CHKERRQ(ierr);
  ierr = MatGetOption(bqnk->B, MAT_SPD, &bqnk->is_spd);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode TaoView_BQNK(Tao tao, PetscViewer viewer)
{
  TAO_BNK        *bnk = (TAO_BNK*)tao->data;
  TAO_BQNK       *bqnk = (TAO_BQNK*)bnk->ctx;
  PetscErrorCode ierr;
  PetscBool      isascii;

  PetscFunctionBegin;
  ierr = TaoView_BNK(tao, viewer);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&isascii);CHKERRQ(ierr);
  if (isascii) {
    ierr = PetscViewerPushFormat(viewer, PETSC_VIEWER_ASCII_INFO);CHKERRQ(ierr);
    ierr = MatView(bqnk->B, viewer);CHKERRQ(ierr);
    ierr = PetscViewerPopFormat(viewer);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode TaoDestroy_BQNK(Tao tao)
{
  TAO_BNK        *bnk = (TAO_BNK*)tao->data;
  TAO_BQNK       *bqnk = (TAO_BQNK*)bnk->ctx;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatDestroy(&bnk->Hpre_inactive);CHKERRQ(ierr);
  ierr = MatDestroy(&bnk->H_inactive);CHKERRQ(ierr);
  ierr = MatDestroy(&bqnk->B);CHKERRQ(ierr);
  ierr = PetscFree(bnk->ctx);CHKERRQ(ierr);
  ierr = TaoDestroy_BNK(tao);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PETSC_INTERN PetscErrorCode TaoCreate_BQNK(Tao tao)
{
  TAO_BNK        *bnk;
  TAO_BQNK       *bqnk;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = TaoCreate_BNK(tao);CHKERRQ(ierr);
  tao->ops->solve = TaoSolve_BQNK;
  tao->ops->setfromoptions = TaoSetFromOptions_BQNK;
  tao->ops->destroy = TaoDestroy_BQNK;
  tao->ops->view = TaoView_BQNK;
  tao->ops->setup = TaoSetUp_BQNK;

  bnk = (TAO_BNK *)tao->data;
  bnk->computehessian = TaoBQNKComputeHessian;
  bnk->computestep = TaoBQNKComputeStep;
  bnk->init_type = BNK_INIT_DIRECTION;

  ierr = PetscNewLog(tao,&bqnk);CHKERRQ(ierr);
  bnk->ctx = (void*)bqnk;
  bqnk->is_spd = PETSC_TRUE;

  ierr = MatCreate(PetscObjectComm((PetscObject)tao), &bqnk->B);CHKERRQ(ierr);
  ierr = PetscObjectIncrementTabLevel((PetscObject)bqnk->B, (PetscObject)tao, 1);CHKERRQ(ierr);
  ierr = MatSetType(bqnk->B, MATLMVMSR1);CHKERRQ(ierr);
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
  PetscErrorCode ierr;
  PetscBool      flg = PETSC_FALSE;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompareAny((PetscObject)tao, &flg, TAOBQNLS, TAOBQNKLS, TAOBQNKTR, TAOBQNKTL, "");CHKERRQ(ierr);
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
  PetscErrorCode ierr;
  PetscBool      flg = PETSC_FALSE;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompareAny((PetscObject)tao, &flg, TAOBQNLS, TAOBQNKLS, TAOBQNKTR, TAOBQNKTL, "");CHKERRQ(ierr);
  PetscCheck(flg,PetscObjectComm((PetscObject)tao), PETSC_ERR_ARG_INCOMP, "LMVM Matrix only exists for quasi-Newton algorithms");
  ierr = PetscObjectBaseTypeCompare((PetscObject)B, MATLMVM, &flg);CHKERRQ(ierr);
  PetscCheck(flg,PetscObjectComm((PetscObject)tao), PETSC_ERR_ARG_INCOMP, "Given matrix is not an LMVM matrix");
  if (bqnk->B) {
    ierr = MatDestroy(&bqnk->B);CHKERRQ(ierr);
  }
  ierr = PetscObjectReference((PetscObject)B);CHKERRQ(ierr);
  bqnk->B = B;
  PetscFunctionReturn(0);
}

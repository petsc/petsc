#include <../src/tao/bound/impls/bqnk/bqnk.h>
#include <petscksp.h>

static PetscErrorCode TaoBQNKComputeHessian(Tao tao)
{
  TAO_BNK        *bnk = (TAO_BNK *)tao->data;
  TAO_BQNK       *bqnk = (TAO_BQNK*)bnk->ctx;
  PetscErrorCode ierr;

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
  ierr = MatLMVMUpdate(tao->hessian, tao->solution, bnk->unprojected_gradient);CHKERRQ(ierr);
  ierr = MatLMVMResetShift(tao->hessian);CHKERRQ(ierr);
  /* Prepare the reduced sub-matrices for the inactive set */
  ierr = MatDestroy(&bnk->H_inactive);CHKERRQ(ierr);
  if (bnk->active_idx) {
    ierr = MatCreateSubMatrixVirtual(tao->hessian, bnk->inactive_idx, bnk->inactive_idx, &bnk->H_inactive);CHKERRQ(ierr);
    if (bqnk->pc) {
      ierr = PCLMVMSetIS(bqnk->pc, bnk->inactive_idx);CHKERRQ(ierr);
    }
  } else {
    ierr = PetscObjectReference((PetscObject)tao->hessian);CHKERRQ(ierr);
    bnk->H_inactive = tao->hessian;
    if (bqnk->pc) {
      ierr = PCLMVMClearIS(bqnk->pc);CHKERRQ(ierr);
    }
  }
  ierr = MatDestroy(&bnk->Hpre_inactive);CHKERRQ(ierr);
  ierr = PetscObjectReference((PetscObject)bnk->H_inactive);CHKERRQ(ierr);
  bnk->Hpre_inactive = bnk->H_inactive;
  PetscFunctionReturn(0);
}

static PetscErrorCode TaoBQNKComputeStep(Tao tao, PetscBool shift, KSPConvergedReason *ksp_reason)
{
  TAO_BNK        *bnk = (TAO_BNK *)tao->data;
  TAO_BQNK       *bqnk = (TAO_BQNK*)bnk->ctx;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = TaoBNKComputeStep(tao, shift, ksp_reason);CHKERRQ(ierr);
  if (*ksp_reason < 0) {
    /* Krylov solver failed to converge so reset the LMVM matrix */
    ierr = MatLMVMReset(bqnk->B, PETSC_FALSE);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode TaoSetUp_BQNK(Tao tao)
{
  TAO_BNK        *bnk = (TAO_BNK *)tao->data;
  TAO_BQNK       *bqnk = (TAO_BQNK*)bnk->ctx;
  PetscErrorCode ierr;
  PetscInt       n, N;
  PetscBool      is_lmvm, is_sym, is_sr1;

  PetscFunctionBegin;
  ierr = TaoSetUp_BNK(tao);CHKERRQ(ierr);
  ierr = VecGetLocalSize(tao->solution,&n);CHKERRQ(ierr);
  ierr = VecGetSize(tao->solution,&N);CHKERRQ(ierr);
  ierr = MatSetSizes(bqnk->B, n, n, N, N);CHKERRQ(ierr);
  ierr = MatLMVMAllocate(bqnk->B,tao->solution,bnk->unprojected_gradient);CHKERRQ(ierr);
  ierr = PetscObjectBaseTypeCompare((PetscObject)bqnk->B, MATLMVM, &is_lmvm);CHKERRQ(ierr);
  if (!is_lmvm) SETERRQ(PetscObjectComm((PetscObject)tao), PETSC_ERR_ARG_INCOMP, "Matrix must be an LMVM-type");
  ierr = MatGetOption(bqnk->B, MAT_SYMMETRIC, &is_sym);CHKERRQ(ierr);
  if (!is_sym) SETERRQ(PetscObjectComm((PetscObject)tao), PETSC_ERR_ARG_INCOMP, "LMVM matrix must be symmetric");
  ierr = PetscObjectTypeCompare((PetscObject)bqnk->B, MATLMVMSR1, &is_sr1);CHKERRQ(ierr);
  ierr = KSPGetPC(tao->ksp, &bqnk->pc);CHKERRQ(ierr);
  if (is_sr1) {
    ierr = PCSetType(bqnk->pc, PCNONE);CHKERRQ(ierr);
    bqnk->pc = NULL;
    bqnk->no_scale = PETSC_TRUE;
  } else {
    ierr = PCSetType(bqnk->pc, PCLMVM);CHKERRQ(ierr);
    ierr = PCLMVMSetMatLMVM(bqnk->pc, bqnk->B);CHKERRQ(ierr);
  }
  if (!bqnk->no_scale) {
    if (!bqnk->Bscale) {
      ierr = MatCreateLMVMDiagBrdn(PetscObjectComm((PetscObject)bqnk->B), n, N, &bqnk->Bscale);CHKERRQ(ierr);
      ierr = MatSetOptionsPrefix(bqnk->Bscale, "tao_bqnk_scale_");CHKERRQ(ierr);
      ierr = MatLMVMAllocate(bqnk->Bscale, tao->solution, bnk->unprojected_gradient);CHKERRQ(ierr);
    }
    ierr = MatLMVMSetJ0(bqnk->B, bqnk->Bscale);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode TaoSetFromOptions_BQNK(PetscOptionItems *PetscOptionsObject, Tao tao)
{
  TAO_BNK        *bnk = (TAO_BNK*)tao->data;
  TAO_BQNK       *bqnk = (TAO_BQNK*)bnk->ctx;
  PetscErrorCode ierr;
  
  PetscFunctionBegin;
  ierr = PetscOptionsHead(PetscOptionsObject,"Quasi-Newton-Krylov method for bound constrained optimization");CHKERRQ(ierr);
  ierr = PetscOptionsBool("-tao_bqnk_no_scale","(developer) disable the diagonal Broyden scaling of the BFGS approximation","",bqnk->no_scale,&bqnk->no_scale,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  ierr = TaoSetFromOptions_BNK(PetscOptionsObject, tao);CHKERRQ(ierr);
  if (bnk->init_type == BNK_INIT_INTERPOLATION) bnk->init_type = BNK_INIT_DIRECTION;
  ierr = MatSetFromOptions(bqnk->B);CHKERRQ(ierr);
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
  if (!bqnk->no_scale) {
    ierr = MatDestroy(&bqnk->Bscale);CHKERRQ(ierr);
  }
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
  bqnk->no_scale = PETSC_FALSE;
  
  ierr = MatCreate(PetscObjectComm((PetscObject)tao), &bqnk->B);CHKERRQ(ierr);
  ierr = PetscObjectIncrementTabLevel((PetscObject)bqnk->B, (PetscObject)tao, 1);CHKERRQ(ierr);
  ierr = MatSetOptionsPrefix(bqnk->B, "tao_bqnk_");CHKERRQ(ierr);
  ierr = MatSetType(bqnk->B, MATLMVMSR1);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
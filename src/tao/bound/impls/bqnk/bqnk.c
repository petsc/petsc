#include <../src/tao/bound/impls/bqnk/bqnk.h>
#include <petscksp.h>

/* Routine for computing the approximate quasi-Newton Hessian */

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
  if (bnk->active_idx) {
    ierr = MatDestroy(&bnk->H_inactive);CHKERRQ(ierr);
    ierr = MatCreateSubMatrixVirtual(tao->hessian, bnk->inactive_idx, bnk->inactive_idx, &bnk->H_inactive);CHKERRQ(ierr);
  } else {
    ierr = MatDestroy(&bnk->H_inactive);CHKERRQ(ierr);
    ierr = PetscObjectReference((PetscObject)tao->hessian);CHKERRQ(ierr);
    bnk->H_inactive = tao->hessian;
  }
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

static PetscErrorCode TaoSetFromOptions_BQNK(PetscOptionItems *PetscOptionsObject, Tao tao)
{
  TAO_BNK        *bnk = (TAO_BNK*)tao->data;
  TAO_BQNK       *bqnk = (TAO_BQNK*)bnk->ctx;
  PetscErrorCode ierr;
  PC             pc;
  PetscBool      is_lmvm, is_sym;
  
  PetscFunctionBegin;
  ierr = TaoSetFromOptions_BNK(PetscOptionsObject, tao);CHKERRQ(ierr);
  /* Make sure the interpolation method is disabled for trust region initialization */
  if (bnk->init_type == BNK_INIT_INTERPOLATION) bnk->init_type = BNK_INIT_DIRECTION;
  /* Make sure the KSP solver is a CG method and the preconditioner is disabled */
  if (!bnk->is_nash && !bnk->is_stcg && !bnk->is_gltr) SETERRQ(PETSC_COMM_SELF,1,"Must use a trust-region CG method for KSP (KSPNASH, KSPSTCG, KSPGLTR)");
  ierr = KSPGetPC(tao->ksp, &pc);CHKERRQ(ierr);
  ierr = PCSetType(pc, PCNONE);CHKERRQ(ierr);
  /* Read mat options and check type and properties */
  ierr = MatSetFromOptions(bqnk->B);CHKERRQ(ierr);
  ierr = PetscObjectBaseTypeCompare((PetscObject)bqnk->B, MATLMVM, &is_lmvm);CHKERRQ(ierr);
  if (!is_lmvm) SETERRQ(PetscObjectComm((PetscObject)tao), PETSC_ERR_ARG_INCOMP, "Matrix must be an LMVM-type");
  ierr = MatGetOption(bqnk->B, MAT_SYMMETRIC, &is_sym);CHKERRQ(ierr);
  if (!is_sym) SETERRQ(PetscObjectComm((PetscObject)tao), PETSC_ERR_ARG_INCOMP, "LMVM matrix must be symmetric");
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
  tao->ops->setfromoptions = TaoSetFromOptions_BQNK;
  tao->ops->destroy = TaoDestroy_BQNK;
  tao->ops->view = TaoView_BQNK;
  
  bnk = (TAO_BNK *)tao->data;
  bnk->computehessian = TaoBQNKComputeHessian;
  bnk->computestep = TaoBQNKComputeStep;
  bnk->init_type = BNK_INIT_DIRECTION;
  
  ierr = PetscNewLog(tao,&bqnk);CHKERRQ(ierr);
  bnk->ctx = (void*)bqnk;
  
  ierr = MatCreate(PetscObjectComm((PetscObject)tao), &bqnk->B);CHKERRQ(ierr);
  ierr = PetscObjectIncrementTabLevel((PetscObject)bqnk->B, (PetscObject)tao, 1);CHKERRQ(ierr);
  ierr = MatSetOptionsPrefix(bqnk->B, "tao_bqnk_");CHKERRQ(ierr);
  ierr = MatSetType(bqnk->B, MATLMVMSR1);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
#include <../src/tao/bound/impls/bqnk/bqnk.h>
#include <petscksp.h>

static const char *BQNK_INIT[64] = {"constant", "direction"};
static const char *BNK_UPDATE[64] = {"step", "reduction", "interpolation"};
static const char *BNK_AS[64] = {"none", "bertsekas"};

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
  if (!is_lmvm) SETERRQ(PetscObjectComm((PetscObject)tao), PETSC_ERR_ARG_INCOMP, "Matrix must be an LMVM-type");
  ierr = MatGetOption(bqnk->B, MAT_SYMMETRIC, &is_sym);CHKERRQ(ierr);
  if (!is_sym) SETERRQ(PetscObjectComm((PetscObject)tao), PETSC_ERR_ARG_INCOMP, "LMVM matrix must be symmetric");
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
  KSPType        ksp_type;

  PetscFunctionBegin;
  ierr = PetscOptionsHead(PetscOptionsObject,"Quasi-Newton-Krylov method for bound constrained optimization");CHKERRQ(ierr);
  ierr = PetscOptionsEList("-tao_bqnk_init_type", "radius initialization type", "", BQNK_INIT, BQNK_INIT_TYPES, BQNK_INIT[bnk->init_type], &bnk->init_type, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEList("-tao_bqnk_update_type", "radius update type", "", BNK_UPDATE, BNK_UPDATE_TYPES, BNK_UPDATE[bnk->update_type], &bnk->update_type, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEList("-tao_bqnk_as_type", "active set estimation method", "", BNK_AS, BNK_AS_TYPES, BNK_AS[bnk->as_type], &bnk->as_type, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_bqnk_sval", "(developer) Hessian perturbation starting value", "", bnk->sval, &bnk->sval,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_bqnk_imin", "(developer) minimum initial Hessian perturbation", "", bnk->imin, &bnk->imin,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_bqnk_imax", "(developer) maximum initial Hessian perturbation", "", bnk->imax, &bnk->imax,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_bqnk_imfac", "(developer) initial merit factor for Hessian perturbation", "", bnk->imfac, &bnk->imfac,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_bqnk_pmin", "(developer) minimum Hessian perturbation", "", bnk->pmin, &bnk->pmin,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_bqnk_pmax", "(developer) maximum Hessian perturbation", "", bnk->pmax, &bnk->pmax,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_bqnk_pgfac", "(developer) Hessian perturbation growth factor", "", bnk->pgfac, &bnk->pgfac,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_bqnk_psfac", "(developer) Hessian perturbation shrink factor", "", bnk->psfac, &bnk->psfac,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_bqnk_pmgfac", "(developer) merit growth factor for Hessian perturbation", "", bnk->pmgfac, &bnk->pmgfac,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_bqnk_pmsfac", "(developer) merit shrink factor for Hessian perturbation", "", bnk->pmsfac, &bnk->pmsfac,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_bqnk_eta1", "(developer) threshold for rejecting step (-tao_bqnk_update_type reduction)", "", bnk->eta1, &bnk->eta1,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_bqnk_eta2", "(developer) threshold for accepting marginal step (-tao_bqnk_update_type reduction)", "", bnk->eta2, &bnk->eta2,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_bqnk_eta3", "(developer) threshold for accepting reasonable step (-tao_bqnk_update_type reduction)", "", bnk->eta3, &bnk->eta3,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_bqnk_eta4", "(developer) threshold for accepting good step (-tao_bqnk_update_type reduction)", "", bnk->eta4, &bnk->eta4,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_bqnk_alpha1", "(developer) radius reduction factor for rejected step (-tao_bqnk_update_type reduction)", "", bnk->alpha1, &bnk->alpha1,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_bqnk_alpha2", "(developer) radius reduction factor for marginally accepted bad step (-tao_bqnk_update_type reduction)", "", bnk->alpha2, &bnk->alpha2,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_bqnk_alpha3", "(developer) radius increase factor for reasonable accepted step (-tao_bqnk_update_type reduction)", "", bnk->alpha3, &bnk->alpha3,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_bqnk_alpha4", "(developer) radius increase factor for good accepted step (-tao_bqnk_update_type reduction)", "", bnk->alpha4, &bnk->alpha4,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_bqnk_alpha5", "(developer) radius increase factor for very good accepted step (-tao_bqnk_update_type reduction)", "", bnk->alpha5, &bnk->alpha5,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_bqnk_nu1", "(developer) threshold for small line-search step length (-tao_bqnk_update_type step)", "", bnk->nu1, &bnk->nu1,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_bqnk_nu2", "(developer) threshold for reasonable line-search step length (-tao_bqnk_update_type step)", "", bnk->nu2, &bnk->nu2,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_bqnk_nu3", "(developer) threshold for large line-search step length (-tao_bqnk_update_type step)", "", bnk->nu3, &bnk->nu3,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_bqnk_nu4", "(developer) threshold for very large line-search step length (-tao_bqnk_update_type step)", "", bnk->nu4, &bnk->nu4,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_bqnk_omega1", "(developer) radius reduction factor for very small line-search step length (-tao_bqnk_update_type step)", "", bnk->omega1, &bnk->omega1,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_bqnk_omega2", "(developer) radius reduction factor for small line-search step length (-tao_bqnk_update_type step)", "", bnk->omega2, &bnk->omega2,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_bqnk_omega3", "(developer) radius factor for decent line-search step length (-tao_bqnk_update_type step)", "", bnk->omega3, &bnk->omega3,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_bqnk_omega4", "(developer) radius increase factor for large line-search step length (-tao_bqnk_update_type step)", "", bnk->omega4, &bnk->omega4,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_bqnk_omega5", "(developer) radius increase factor for very large line-search step length (-tao_bqnk_update_type step)", "", bnk->omega5, &bnk->omega5,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_bqnk_mu1", "(developer) threshold for accepting very good step (-tao_bqnk_update_type interpolation)", "", bnk->mu1, &bnk->mu1,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_bqnk_mu2", "(developer) threshold for accepting good step (-tao_bqnk_update_type interpolation)", "", bnk->mu2, &bnk->mu2,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_bqnk_gamma1", "(developer) radius reduction factor for rejected very bad step (-tao_bqnk_update_type interpolation)", "", bnk->gamma1, &bnk->gamma1,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_bqnk_gamma2", "(developer) radius reduction factor for rejected bad step (-tao_bqnk_update_type interpolation)", "", bnk->gamma2, &bnk->gamma2,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_bqnk_gamma3", "(developer) radius increase factor for accepted good step (-tao_bqnk_update_type interpolation)", "", bnk->gamma3, &bnk->gamma3,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_bqnk_gamma4", "(developer) radius increase factor for accepted very good step (-tao_bqnk_update_type interpolation)", "", bnk->gamma4, &bnk->gamma4,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_bqnk_theta", "(developer) trust region interpolation factor (-tao_bqnk_update_type interpolation)", "", bnk->theta, &bnk->theta,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_bqnk_min_radius", "(developer) lower bound on initial radius", "", bnk->min_radius, &bnk->min_radius,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_bqnk_max_radius", "(developer) upper bound on radius", "", bnk->max_radius, &bnk->max_radius,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_bqnk_epsilon", "(developer) tolerance used when computing actual and predicted reduction", "", bnk->epsilon, &bnk->epsilon,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_bqnk_as_tol", "(developer) initial tolerance used when estimating actively bounded variables", "", bnk->as_tol, &bnk->as_tol,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_bqnk_as_step", "(developer) step length used when estimating actively bounded variables", "", bnk->as_step, &bnk->as_step,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-tao_bqnk_max_cg_its", "number of BNCG iterations to take for each Newton step", "", bnk->max_cg_its, &bnk->max_cg_its,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  ierr = TaoSetFromOptions(bnk->bncg);CHKERRQ(ierr);
  ierr = TaoLineSearchSetFromOptions(tao->linesearch);CHKERRQ(ierr);
  ierr = KSPSetFromOptions(tao->ksp);CHKERRQ(ierr);
  ierr = KSPGetType(tao->ksp,&ksp_type);CHKERRQ(ierr);
  ierr = PetscStrcmp(ksp_type,KSPNASH,&bnk->is_nash);CHKERRQ(ierr);
  ierr = PetscStrcmp(ksp_type,KSPSTCG,&bnk->is_stcg);CHKERRQ(ierr);
  ierr = PetscStrcmp(ksp_type,KSPGLTR,&bnk->is_gltr);CHKERRQ(ierr);
  if (bnk->init_type == BNK_INIT_INTERPOLATION) bnk->init_type = BNK_INIT_DIRECTION;
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
  ierr = KSPSetOptionsPrefix(tao->ksp, "tao_bqnk_");CHKERRQ(ierr);
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
  ierr = MatSetOptionsPrefix(bqnk->B, "tao_bqnk_");CHKERRQ(ierr);
  ierr = MatSetType(bqnk->B, MATLMVMSR1);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode TaoGetLMVMMatrix(Tao tao, Mat *B)
{
  TAO_BNK        *bnk = (TAO_BNK*)tao->data;
  TAO_BQNK       *bqnk = (TAO_BQNK*)bnk->ctx;
  PetscErrorCode ierr;
  PetscBool      is_bqnls, is_bqnkls, is_bqnktr, is_bqnktl;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)tao, TAOBQNLS, &is_bqnls);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)tao, TAOBQNKLS, &is_bqnkls);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)tao, TAOBQNKTR, &is_bqnktr);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)tao, TAOBQNKTL, &is_bqnktl);CHKERRQ(ierr);
  if (!is_bqnls && !is_bqnkls && !is_bqnktr && is_bqnktl) SETERRQ(PetscObjectComm((PetscObject)tao), PETSC_ERR_ARG_INCOMP, "LMVM Matrix only exists for quasi-Newton algorithms");
  *B = bqnk->B;
  PetscFunctionReturn(0);
}

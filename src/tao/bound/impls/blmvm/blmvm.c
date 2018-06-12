#include <../src/tao/bound/impls/bqnk/bqnk.h>

PETSC_EXTERN PetscErrorCode TaoCreate_BQNLS(Tao);

static PetscErrorCode TaoSetFromOptions_BLMVM(PetscOptionItems *PetscOptionsObject,Tao tao)
{
  TAO_BNK        *bnk = (TAO_BNK *)tao->data;
  TAO_BQNK       *bqnk = (TAO_BQNK*)bnk->ctx;
  PetscErrorCode ierr;
  KSPType        ksp_type;
  PetscBool      is_spd;

  PetscFunctionBegin;
  ierr = PetscOptionsHead(PetscOptionsObject,"Quasi-Newton-Krylov method for bound constrained optimization");CHKERRQ(ierr);
  ierr = PetscOptionsEList("-tao_blmvm_as_type", "active set estimation method", "", BNK_AS, BNK_AS_TYPES, BNK_AS[bnk->as_type], &bnk->as_type, 0);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_blmvm_epsilon", "(developer) tolerance used when computing actual and predicted reduction", "", bnk->epsilon, &bnk->epsilon,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_blmvm_as_tol", "(developer) initial tolerance used when estimating actively bounded variables", "", bnk->as_tol, &bnk->as_tol,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_blmvm_as_step", "(developer) step length used when estimating actively bounded variables", "", bnk->as_step, &bnk->as_step,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-tao_blmvm_max_cg_its", "number of BNCG iterations to take for each Newton step", "", bnk->max_cg_its, &bnk->max_cg_its,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  ierr = TaoSetFromOptions(bnk->bncg);CHKERRQ(ierr);
  ierr = TaoLineSearchSetFromOptions(tao->linesearch);CHKERRQ(ierr);
  ierr = KSPSetFromOptions(tao->ksp);CHKERRQ(ierr);
  ierr = KSPGetType(tao->ksp,&ksp_type);CHKERRQ(ierr);
  bnk->is_nash = bnk->is_gltr = bnk->is_stcg = PETSC_FALSE;
  ierr = MatSetFromOptions(bqnk->B);CHKERRQ(ierr);
  ierr = MatGetOption(bqnk->B, MAT_SPD, &is_spd);CHKERRQ(ierr);
  if (!is_spd) SETERRQ(PetscObjectComm((PetscObject)tao), PETSC_ERR_ARG_INCOMP, "LMVM matrix must be symmetric positive-definite");
  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode TaoCreate_BLMVM(Tao tao)
{
  PetscErrorCode ierr;
  
  PetscFunctionBegin;
  ierr = PetscPrintf(PetscObjectComm((PetscObject)tao), "BLMVM is deprecated and will be removed in a future PETSc/TAO release. Please use BQNLS instead.\n");CHKERRQ(ierr);
  ierr = TaoCreate_BQNLS(tao);CHKERRQ(ierr);
  tao->ops->setfromoptions = TaoSetFromOptions_BLMVM;
  PetscFunctionReturn(0);
}
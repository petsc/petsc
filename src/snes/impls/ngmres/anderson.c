#include <../src/snes/impls/ngmres/snesngmres.h> /*I "petscsnes.h" I*/

PETSC_EXTERN PetscErrorCode SNESDestroy_NGMRES(SNES);
PETSC_EXTERN PetscErrorCode SNESReset_NGMRES(SNES);
PETSC_EXTERN PetscErrorCode SNESSetUp_NGMRES(SNES);
PETSC_EXTERN PetscErrorCode SNESView_NGMRES(SNES,PetscViewer);


#undef __FUNCT__
#define __FUNCT__ "SNESSetFromOptions_Anderson"
PetscErrorCode SNESSetFromOptions_Anderson(SNES snes)
{
  SNES_NGMRES   *ngmres = (SNES_NGMRES *) snes->data;
  PetscErrorCode ierr;
  PetscBool      debug;
  SNESLineSearch linesearch;

  PetscFunctionBegin;
  ierr = PetscOptionsHead("SNES NGMRES options");CHKERRQ(ierr);
  ierr = PetscOptionsInt("-snes_anderson_m","Number of directions","SNES",ngmres->msize,&ngmres->msize,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-snes_anderson_beta","Number of directions","SNES",ngmres->andersonBeta,&ngmres->andersonBeta,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-snes_anderson_monitor","Monitor actions of NGMRES","SNES",ngmres->monitor ? PETSC_TRUE: PETSC_FALSE,&debug,PETSC_NULL);CHKERRQ(ierr);
  if (debug) {
    ngmres->monitor = PETSC_VIEWER_STDOUT_(((PetscObject)snes)->comm);CHKERRQ(ierr);
  }
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  /* set the default type of the line search if the user hasn't already. */
  if (!snes->linesearch) {
    ierr = SNESGetSNESLineSearch(snes,&linesearch);CHKERRQ(ierr);
    ierr = SNESLineSearchSetType(linesearch,SNESLINESEARCHBASIC);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESSolve_Anderson"
PetscErrorCode SNESSolve_Anderson(SNES snes)
{
  SNES_NGMRES        *ngmres = (SNES_NGMRES *) snes->data;
  /* present solution, residual, and preconditioned residual */
  Vec                 X,F,B;

  /* candidate linear combination answers */
  Vec                 XA,FA,XM,FM,FPC;

  /* coefficients and RHS to the minimization problem */
  PetscReal           fnorm,fMnorm;
  PetscInt            k,k_restart,l,ivec;

  SNESConvergedReason reason;
  PetscErrorCode      ierr;

  PetscFunctionBegin;
  /* variable initialization */
  snes->reason  = SNES_CONVERGED_ITERATING;
  X             = snes->vec_sol;
  F             = snes->vec_func;
  B             = snes->vec_rhs;
  XA            = snes->vec_sol_update;
  FA            = snes->work[0];

  /* work for the line search */
  XM            = snes->work[3];
  FM            = snes->work[4];

  ierr = PetscObjectTakeAccess(snes);CHKERRQ(ierr);
  snes->iter = 0;
  snes->norm = 0.;
  ierr = PetscObjectGrantAccess(snes);CHKERRQ(ierr);

  /* initialization */

  /* r = F(x) */
  if (!snes->vec_func_init_set) {
    ierr = SNESComputeFunction(snes,X,F);CHKERRQ(ierr);
    if (snes->domainerror) {
      snes->reason = SNES_DIVERGED_FUNCTION_DOMAIN;
      PetscFunctionReturn(0);
    }
  } else {
    snes->vec_func_init_set = PETSC_FALSE;
  }

  if (!snes->norm_init_set) {
    ierr = VecNorm(F,NORM_2,&fnorm);CHKERRQ(ierr);
    if (PetscIsInfOrNanReal(fnorm)) SETERRQ(((PetscObject)snes)->comm,PETSC_ERR_FP,"Infinite or not-a-number generated in function evaluation");
  } else {
    fnorm = snes->norm_init;
    snes->norm_init_set = PETSC_FALSE;
  }
  ierr = PetscObjectTakeAccess(snes);CHKERRQ(ierr);
  snes->norm = fnorm;
  ierr = PetscObjectGrantAccess(snes);CHKERRQ(ierr);
  SNESLogConvHistory(snes,fnorm,0);
  ierr = SNESMonitor(snes,0,fnorm);CHKERRQ(ierr);
  ierr = (*snes->ops->converged)(snes,0,0.0,0.0,fnorm,&snes->reason,snes->cnvP);CHKERRQ(ierr);
  if (snes->reason) PetscFunctionReturn(0);

  k_restart = 0;
  l = 0;
  for (k=1; k < snes->max_its+1; k++) {
    /* select which vector of the stored subspace will be updated */
    ivec = k_restart % ngmres->msize;
    if (snes->pc && snes->pcside == PC_RIGHT) {
      ierr = VecCopy(X,XM);CHKERRQ(ierr);
      ierr = SNESSetInitialFunction(snes->pc,F);CHKERRQ(ierr);
      ierr = SNESSetInitialFunctionNorm(snes->pc,fnorm);CHKERRQ(ierr);

      ierr = PetscLogEventBegin(SNES_NPCSolve,snes->pc,XM,B,0);CHKERRQ(ierr);
      ierr = SNESSolve(snes->pc,B,XM);CHKERRQ(ierr);
      ierr = PetscLogEventEnd(SNES_NPCSolve,snes->pc,XM,B,0);CHKERRQ(ierr);

      ierr = SNESGetConvergedReason(snes->pc,&reason);CHKERRQ(ierr);
      if (reason < 0 && reason != SNES_DIVERGED_MAX_IT) {
        snes->reason = SNES_DIVERGED_INNER;
        PetscFunctionReturn(0);
      }
      ierr = SNESGetFunction(snes->pc,&FPC,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);
      ierr = VecCopy(FPC,FM);CHKERRQ(ierr);
      ierr = SNESGetFunctionNorm(snes->pc,&fMnorm);CHKERRQ(ierr);
      if (ngmres->andersonBeta != 1.0) {
        VecAXPBY(XM,(1.0 - ngmres->andersonBeta),ngmres->andersonBeta,X);CHKERRQ(ierr);
      }
    } else {
      ierr = VecCopy(F,FM);CHKERRQ(ierr);
      ierr = VecCopy(X,XM);CHKERRQ(ierr);
      ierr = VecAXPY(XM,-ngmres->andersonBeta,FM);CHKERRQ(ierr);
      fMnorm = fnorm;
    }

    ierr = SNESNGMRESFormCombinedSolution_Private(snes,l,XM,FM,fMnorm,X,XA,FA);

    if (l < ngmres->msize)l++;
    k_restart++;
    ierr = SNESNGMRESUpdateSubspace_Private(snes,ivec,l,FM,fnorm,XM);

    ierr = VecNorm(FA,NORM_2,&fnorm);CHKERRQ(ierr);
    ierr = VecCopy(XA,X);CHKERRQ(ierr);
    ierr = VecCopy(FA,F);CHKERRQ(ierr);

    ierr = PetscObjectTakeAccess(snes);CHKERRQ(ierr);
    snes->iter = k;
    snes->norm = fnorm;
    ierr = PetscObjectGrantAccess(snes);CHKERRQ(ierr);
    SNESLogConvHistory(snes,snes->norm,snes->iter);
    ierr = SNESMonitor(snes,snes->iter,snes->norm);CHKERRQ(ierr);
    ierr = (*snes->ops->converged)(snes,snes->iter,0.0,0.0,fnorm,&snes->reason,snes->cnvP);CHKERRQ(ierr);
    if (snes->reason) PetscFunctionReturn(0);
  }
  snes->reason = SNES_DIVERGED_MAX_IT;
  PetscFunctionReturn(0);
}

/*MC
  SNESAnderson - Anderson Mixing method.

   Level: beginner

   Options Database:
+  -snes_anderson_m                - Number of stored previous solutions and residuals
.  -snes_anderson_beta             - Relaxation parameter; X_{update} = X + \beta F
-  -snes_anderson_monitor          - Prints relevant information about the ngmres iteration

   Notes:

   The Anderson Mixing method combines m previous solutions into a minimum-residual solution by solving a small linearized
   optimization problem at each iteration.

   References:

    "D. G. Anderson. Iterative procedures for nonlinear integral equations.
    J. Assoc. Comput. Mach., 12:547–560, 1965."

.seealso: SNESCreate(), SNES, SNESSetType(), SNESType (for list of available types)
M*/

EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "SNESCreate_Anderson"
PetscErrorCode SNESCreate_Anderson(SNES snes)
{
  SNES_NGMRES   *ngmres;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  snes->ops->destroy        = SNESDestroy_NGMRES;
  snes->ops->setup          = SNESSetUp_NGMRES;
  snes->ops->setfromoptions = SNESSetFromOptions_Anderson;
  snes->ops->view           = SNESView_NGMRES;
  snes->ops->solve          = SNESSolve_Anderson;
  snes->ops->reset          = SNESReset_NGMRES;

  snes->usespc          = PETSC_TRUE;
  snes->usesksp         = PETSC_FALSE;

  ierr = PetscNewLog(snes,SNES_NGMRES,&ngmres);CHKERRQ(ierr);
  snes->data = (void*) ngmres;
  ngmres->msize = 30;

  if (!snes->tolerancesset) {
    snes->max_funcs = 30000;
    snes->max_its   = 10000;
  }

  ngmres->additive_linesearch = PETSC_NULL;

  ngmres->restart_it = 2;
  ngmres->restart_periodic = 30;
  ngmres->gammaA     = 2.0;
  ngmres->gammaC     = 2.0;
  ngmres->deltaB     = 0.9;
  ngmres->epsilonB   = 0.1;

  ngmres->andersonBeta = 1.0;

  PetscFunctionReturn(0);
}
EXTERN_C_END

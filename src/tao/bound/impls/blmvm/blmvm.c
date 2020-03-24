#include <petsctaolinesearch.h>      /*I "petsctaolinesearch.h" I*/
#include <../src/tao/unconstrained/impls/lmvm/lmvm.h>
#include <../src/tao/bound/impls/blmvm/blmvm.h>

/*------------------------------------------------------------*/
static PetscErrorCode TaoSolve_BLMVM(Tao tao)
{
  PetscErrorCode               ierr;
  TAO_BLMVM                    *blmP = (TAO_BLMVM *)tao->data;
  TaoLineSearchConvergedReason ls_status = TAOLINESEARCH_CONTINUE_ITERATING;
  PetscReal                    f, fold, gdx, gnorm, gnorm2;
  PetscReal                    stepsize = 1.0,delta;

  PetscFunctionBegin;
  /*  Project initial point onto bounds */
  ierr = TaoComputeVariableBounds(tao);CHKERRQ(ierr);
  ierr = VecMedian(tao->XL,tao->solution,tao->XU,tao->solution);CHKERRQ(ierr);
  ierr = TaoLineSearchSetVariableBounds(tao->linesearch,tao->XL,tao->XU);CHKERRQ(ierr);


  /* Check convergence criteria */
  ierr = TaoComputeObjectiveAndGradient(tao, tao->solution,&f,blmP->unprojected_gradient);CHKERRQ(ierr);
  ierr = VecBoundGradientProjection(blmP->unprojected_gradient,tao->solution, tao->XL,tao->XU,tao->gradient);CHKERRQ(ierr);

  ierr = TaoGradientNorm(tao, tao->gradient,NORM_2,&gnorm);CHKERRQ(ierr);
  if (PetscIsInfOrNanReal(f) || PetscIsInfOrNanReal(gnorm)) SETERRQ(PetscObjectComm((PetscObject)tao),PETSC_ERR_USER, "User provided compute function generated Inf or NaN");

  tao->reason = TAO_CONTINUE_ITERATING;
  ierr = TaoLogConvergenceHistory(tao,f,gnorm,0.0,tao->ksp_its);CHKERRQ(ierr);
  ierr = TaoMonitor(tao,tao->niter,f,gnorm,0.0,stepsize);CHKERRQ(ierr);
  ierr = (*tao->ops->convergencetest)(tao,tao->cnvP);CHKERRQ(ierr);
  if (tao->reason != TAO_CONTINUE_ITERATING) PetscFunctionReturn(0);

  /* Set counter for gradient/reset steps */
  if (!blmP->recycle) {
    blmP->grad = 0;
    blmP->reset = 0;
    ierr = MatLMVMReset(blmP->M, PETSC_FALSE);CHKERRQ(ierr);
  }

  /* Have not converged; continue with Newton method */
  while (tao->reason == TAO_CONTINUE_ITERATING) {
    /* Call general purpose update function */
    if (tao->ops->update) {
      ierr = (*tao->ops->update)(tao, tao->niter, tao->user_update);CHKERRQ(ierr);
    }
    /* Compute direction */
    gnorm2 = gnorm*gnorm;
    if (gnorm2 == 0.0) gnorm2 = PETSC_MACHINE_EPSILON;
    if (f == 0.0) {
      delta = 2.0 / gnorm2;
    } else {
      delta = 2.0 * PetscAbsScalar(f) / gnorm2;
    }
    ierr = MatLMVMSymBroydenSetDelta(blmP->M, delta);CHKERRQ(ierr);
    ierr = MatLMVMUpdate(blmP->M, tao->solution, tao->gradient);CHKERRQ(ierr);
    ierr = MatSolve(blmP->M, blmP->unprojected_gradient, tao->stepdirection);CHKERRQ(ierr);
    ierr = VecBoundGradientProjection(tao->stepdirection,tao->solution,tao->XL,tao->XU,tao->gradient);CHKERRQ(ierr);

    /* Check for success (descent direction) */
    ierr = VecDot(blmP->unprojected_gradient, tao->gradient, &gdx);CHKERRQ(ierr);
    if (gdx <= 0) {
      /* Step is not descent or solve was not successful
         Use steepest descent direction (scaled) */
      ++blmP->grad;

      ierr = MatLMVMReset(blmP->M, PETSC_FALSE);CHKERRQ(ierr);
      ierr = MatLMVMUpdate(blmP->M, tao->solution, blmP->unprojected_gradient);CHKERRQ(ierr);
      ierr = MatSolve(blmP->M,blmP->unprojected_gradient, tao->stepdirection);CHKERRQ(ierr);
    }
    ierr = VecScale(tao->stepdirection,-1.0);CHKERRQ(ierr);

    /* Perform the linesearch */
    fold = f;
    ierr = VecCopy(tao->solution, blmP->Xold);CHKERRQ(ierr);
    ierr = VecCopy(blmP->unprojected_gradient, blmP->Gold);CHKERRQ(ierr);
    ierr = TaoLineSearchSetInitialStepLength(tao->linesearch,1.0);CHKERRQ(ierr);
    ierr = TaoLineSearchApply(tao->linesearch, tao->solution, &f, blmP->unprojected_gradient, tao->stepdirection, &stepsize, &ls_status);CHKERRQ(ierr);
    ierr = TaoAddLineSearchCounts(tao);CHKERRQ(ierr);

    if (ls_status != TAOLINESEARCH_SUCCESS && ls_status != TAOLINESEARCH_SUCCESS_USER) {
      /* Linesearch failed
         Reset factors and use scaled (projected) gradient step */
      ++blmP->reset;

      f = fold;
      ierr = VecCopy(blmP->Xold, tao->solution);CHKERRQ(ierr);
      ierr = VecCopy(blmP->Gold, blmP->unprojected_gradient);CHKERRQ(ierr);

      ierr = MatLMVMReset(blmP->M, PETSC_FALSE);CHKERRQ(ierr);
      ierr = MatLMVMUpdate(blmP->M, tao->solution, blmP->unprojected_gradient);CHKERRQ(ierr);
      ierr = MatSolve(blmP->M, blmP->unprojected_gradient, tao->stepdirection);CHKERRQ(ierr);
      ierr = VecScale(tao->stepdirection, -1.0);CHKERRQ(ierr);

      /* This may be incorrect; linesearch has values for stepmax and stepmin
         that should be reset. */
      ierr = TaoLineSearchSetInitialStepLength(tao->linesearch,1.0);CHKERRQ(ierr);
      ierr = TaoLineSearchApply(tao->linesearch,tao->solution,&f, blmP->unprojected_gradient, tao->stepdirection,  &stepsize, &ls_status);CHKERRQ(ierr);
      ierr = TaoAddLineSearchCounts(tao);CHKERRQ(ierr);

      if (ls_status != TAOLINESEARCH_SUCCESS && ls_status != TAOLINESEARCH_SUCCESS_USER) {
        tao->reason = TAO_DIVERGED_LS_FAILURE;
        break;
      }
    }

    /* Check for converged */
    ierr = VecBoundGradientProjection(blmP->unprojected_gradient, tao->solution, tao->XL, tao->XU, tao->gradient);CHKERRQ(ierr);
    ierr = TaoGradientNorm(tao, tao->gradient, NORM_2, &gnorm);CHKERRQ(ierr);
    if (PetscIsInfOrNanReal(f) || PetscIsInfOrNanReal(gnorm)) SETERRQ(PetscObjectComm((PetscObject)tao),PETSC_ERR_USER, "User provided compute function generated Not-a-Number");
    tao->niter++;
    ierr = TaoLogConvergenceHistory(tao,f,gnorm,0.0,tao->ksp_its);CHKERRQ(ierr);
    ierr = TaoMonitor(tao,tao->niter,f,gnorm,0.0,stepsize);CHKERRQ(ierr);
    ierr = (*tao->ops->convergencetest)(tao,tao->cnvP);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode TaoSetup_BLMVM(Tao tao)
{
  TAO_BLMVM      *blmP = (TAO_BLMVM *)tao->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* Existence of tao->solution checked in TaoSetup() */
  ierr = VecDuplicate(tao->solution,&blmP->Xold);CHKERRQ(ierr);
  ierr = VecDuplicate(tao->solution,&blmP->Gold);CHKERRQ(ierr);
  ierr = VecDuplicate(tao->solution, &blmP->unprojected_gradient);CHKERRQ(ierr);

  if (!tao->stepdirection) {
    ierr = VecDuplicate(tao->solution, &tao->stepdirection);CHKERRQ(ierr);
  }
  if (!tao->gradient) {
    ierr = VecDuplicate(tao->solution,&tao->gradient);CHKERRQ(ierr);
  }
  if (!tao->XL) {
    ierr = VecDuplicate(tao->solution,&tao->XL);CHKERRQ(ierr);
    ierr = VecSet(tao->XL,PETSC_NINFINITY);CHKERRQ(ierr);
  }
  if (!tao->XU) {
    ierr = VecDuplicate(tao->solution,&tao->XU);CHKERRQ(ierr);
    ierr = VecSet(tao->XU,PETSC_INFINITY);CHKERRQ(ierr);
  }
  /* Allocate matrix for the limited memory approximation */
  ierr = MatLMVMAllocate(blmP->M,tao->solution,blmP->unprojected_gradient);CHKERRQ(ierr);

  /* If the user has set a matrix to solve as the initial H0, set the options prefix here, and set up the KSP */
  if (blmP->H0) {
    ierr = MatLMVMSetJ0(blmP->M, blmP->H0);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------- */
static PetscErrorCode TaoDestroy_BLMVM(Tao tao)
{
  TAO_BLMVM      *blmP = (TAO_BLMVM *)tao->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (tao->setupcalled) {
    ierr = VecDestroy(&blmP->unprojected_gradient);CHKERRQ(ierr);
    ierr = VecDestroy(&blmP->Xold);CHKERRQ(ierr);
    ierr = VecDestroy(&blmP->Gold);CHKERRQ(ierr);
  }
  ierr = MatDestroy(&blmP->M);CHKERRQ(ierr);
  if (blmP->H0) {
    PetscObjectDereference((PetscObject)blmP->H0);
  }
  ierr = PetscFree(tao->data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/
static PetscErrorCode TaoSetFromOptions_BLMVM(PetscOptionItems* PetscOptionsObject,Tao tao)
{
  TAO_BLMVM      *blmP = (TAO_BLMVM *)tao->data;
  PetscErrorCode ierr;
  PetscBool      is_spd;

  PetscFunctionBegin;
  ierr = PetscOptionsHead(PetscOptionsObject,"Limited-memory variable-metric method for bound constrained optimization");CHKERRQ(ierr);
  ierr = PetscOptionsBool("-tao_blmvm_recycle","enable recycling of the BFGS matrix between subsequent TaoSolve() calls","",blmP->recycle,&blmP->recycle,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  ierr = TaoLineSearchSetFromOptions(tao->linesearch);CHKERRQ(ierr);
  ierr = MatSetFromOptions(blmP->M);CHKERRQ(ierr);
  ierr = MatGetOption(blmP->M, MAT_SPD, &is_spd);CHKERRQ(ierr);
  if (!is_spd) SETERRQ(PetscObjectComm((PetscObject)tao), PETSC_ERR_ARG_INCOMP, "LMVM matrix must be symmetric positive-definite");
  PetscFunctionReturn(0);
}


/*------------------------------------------------------------*/
static PetscErrorCode TaoView_BLMVM(Tao tao, PetscViewer viewer)
{
  TAO_BLMVM      *lmP = (TAO_BLMVM *)tao->data;
  PetscBool      isascii;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &isascii);CHKERRQ(ierr);
  if (isascii) {
    ierr = PetscViewerASCIIPrintf(viewer, "Gradient steps: %D\n", lmP->grad);CHKERRQ(ierr);
    ierr = PetscViewerPushFormat(viewer, PETSC_VIEWER_ASCII_INFO);CHKERRQ(ierr);
    ierr = MatView(lmP->M, viewer);CHKERRQ(ierr);
    ierr = PetscViewerPopFormat(viewer);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode TaoComputeDual_BLMVM(Tao tao, Vec DXL, Vec DXU)
{
  TAO_BLMVM      *blm = (TAO_BLMVM *) tao->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tao,TAO_CLASSID,1);
  PetscValidHeaderSpecific(DXL,VEC_CLASSID,2);
  PetscValidHeaderSpecific(DXU,VEC_CLASSID,3);
  if (!tao->gradient || !blm->unprojected_gradient) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ORDER,"Dual variables don't exist yet or no longer exist.\n");

  ierr = VecCopy(tao->gradient,DXL);CHKERRQ(ierr);
  ierr = VecAXPY(DXL,-1.0,blm->unprojected_gradient);CHKERRQ(ierr);
  ierr = VecSet(DXU,0.0);CHKERRQ(ierr);
  ierr = VecPointwiseMax(DXL,DXL,DXU);CHKERRQ(ierr);

  ierr = VecCopy(blm->unprojected_gradient,DXU);CHKERRQ(ierr);
  ierr = VecAXPY(DXU,-1.0,tao->gradient);CHKERRQ(ierr);
  ierr = VecAXPY(DXU,1.0,DXL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------- */
/*MC
  TAOBLMVM - Bounded limited memory variable metric is a quasi-Newton method
         for nonlinear minimization with bound constraints. It is an extension
         of TAOLMVM

  Options Database Keys:
.     -tao_lmm_recycle - enable recycling of LMVM information between subsequent TaoSolve calls

  Level: beginner
M*/
PETSC_EXTERN PetscErrorCode TaoCreate_BLMVM(Tao tao)
{
  TAO_BLMVM      *blmP;
  const char     *morethuente_type = TAOLINESEARCHMT;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  tao->ops->setup = TaoSetup_BLMVM;
  tao->ops->solve = TaoSolve_BLMVM;
  tao->ops->view = TaoView_BLMVM;
  tao->ops->setfromoptions = TaoSetFromOptions_BLMVM;
  tao->ops->destroy = TaoDestroy_BLMVM;
  tao->ops->computedual = TaoComputeDual_BLMVM;

  ierr = PetscNewLog(tao,&blmP);CHKERRQ(ierr);
  blmP->H0 = NULL;
  blmP->recycle = PETSC_FALSE;
  tao->data = (void*)blmP;

  /* Override default settings (unless already changed) */
  if (!tao->max_it_changed) tao->max_it = 2000;
  if (!tao->max_funcs_changed) tao->max_funcs = 4000;

  ierr = TaoLineSearchCreate(((PetscObject)tao)->comm, &tao->linesearch);CHKERRQ(ierr);
  ierr = PetscObjectIncrementTabLevel((PetscObject)tao->linesearch, (PetscObject)tao, 1);CHKERRQ(ierr);
  ierr = TaoLineSearchSetType(tao->linesearch, morethuente_type);CHKERRQ(ierr);
  ierr = TaoLineSearchUseTaoRoutines(tao->linesearch,tao);CHKERRQ(ierr);
  ierr = TaoLineSearchSetOptionsPrefix(tao->linesearch,tao->hdr.prefix);CHKERRQ(ierr);
  
  ierr = KSPInitializePackage();CHKERRQ(ierr);
  ierr = MatCreate(((PetscObject)tao)->comm, &blmP->M);CHKERRQ(ierr);
  ierr = MatSetType(blmP->M, MATLMVMBFGS);CHKERRQ(ierr);
  ierr = PetscObjectIncrementTabLevel((PetscObject)blmP->M, (PetscObject)tao, 1);CHKERRQ(ierr);
  ierr = MatSetOptionsPrefix(blmP->M, "tao_blmvm_");CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
  TaoLMVMRecycle - Enable/disable recycling of the QN history between subsequent TaoSolve calls.

  Input Parameters:
+  tao  - the Tao solver context
-  flg - Boolean flag for recycling (PETSC_TRUE or PETSC_FALSE)

  Level: intermediate
@*/
PetscErrorCode TaoLMVMRecycle(Tao tao, PetscBool flg)
{
  TAO_LMVM       *lmP;
  TAO_BLMVM      *blmP;
  TaoType        type;
  PetscBool      is_lmvm, is_blmvm;
  PetscErrorCode ierr;
  
  PetscFunctionBegin;
  ierr = TaoGetType(tao, &type);CHKERRQ(ierr);
  ierr = PetscStrcmp(type, TAOLMVM,  &is_lmvm);CHKERRQ(ierr);
  ierr = PetscStrcmp(type, TAOBLMVM, &is_blmvm);CHKERRQ(ierr);
  
  if (is_lmvm) {
    lmP = (TAO_LMVM *)tao->data;
    lmP->recycle = flg;
  } else if (is_blmvm) {
    blmP = (TAO_BLMVM *)tao->data;
    blmP->recycle = flg;
  } else SETERRQ(PetscObjectComm((PetscObject)tao), PETSC_ERR_ARG_WRONGSTATE, "This routine applies to TAO_LMVM and TAO_BLMVM.");
  PetscFunctionReturn(0);
}

/*@
  TaoLMVMSetH0 - Set the initial Hessian for the QN approximation

  Input Parameters:
+  tao  - the Tao solver context
-  H0 - Mat object for the initial Hessian

  Level: advanced

.seealso: TaoLMVMGetH0(), TaoLMVMGetH0KSP()
@*/
PetscErrorCode TaoLMVMSetH0(Tao tao, Mat H0)
{
  TAO_LMVM       *lmP;
  TAO_BLMVM      *blmP;
  TaoType        type;
  PetscBool      is_lmvm, is_blmvm;
  PetscErrorCode ierr;
  
  PetscFunctionBegin;
  ierr = TaoGetType(tao, &type);CHKERRQ(ierr);
  ierr = PetscStrcmp(type, TAOLMVM,  &is_lmvm);CHKERRQ(ierr);
  ierr = PetscStrcmp(type, TAOBLMVM, &is_blmvm);CHKERRQ(ierr);

  if (is_lmvm) {
    lmP = (TAO_LMVM *)tao->data;
    ierr = PetscObjectReference((PetscObject)H0);CHKERRQ(ierr);
    lmP->H0 = H0;
  } else if (is_blmvm) {
    blmP = (TAO_BLMVM *)tao->data;
    ierr = PetscObjectReference((PetscObject)H0);CHKERRQ(ierr);
    blmP->H0 = H0;
  } else SETERRQ(PetscObjectComm((PetscObject)tao), PETSC_ERR_ARG_WRONGSTATE, "This routine applies to TAO_LMVM and TAO_BLMVM.");
  PetscFunctionReturn(0);
}

/*@
  TaoLMVMGetH0 - Get the matrix object for the QN initial Hessian

  Input Parameters:
.  tao  - the Tao solver context

  Output Parameters:
.  H0 - Mat object for the initial Hessian

  Level: advanced

.seealso: TaoLMVMSetH0(), TaoLMVMGetH0KSP()
@*/
PetscErrorCode TaoLMVMGetH0(Tao tao, Mat *H0)
{
  TAO_LMVM       *lmP;
  TAO_BLMVM      *blmP;
  TaoType        type;
  PetscBool      is_lmvm, is_blmvm;
  Mat            M;

  PetscErrorCode ierr;
  
  PetscFunctionBegin;
  ierr = TaoGetType(tao, &type);CHKERRQ(ierr);
  ierr = PetscStrcmp(type, TAOLMVM,  &is_lmvm);CHKERRQ(ierr);
  ierr = PetscStrcmp(type, TAOBLMVM, &is_blmvm);CHKERRQ(ierr);

  if (is_lmvm) {
    lmP = (TAO_LMVM *)tao->data;
    M = lmP->M;
  } else if (is_blmvm) {
    blmP = (TAO_BLMVM *)tao->data;
    M = blmP->M;
  } else SETERRQ(PetscObjectComm((PetscObject)tao), PETSC_ERR_ARG_WRONGSTATE, "This routine applies to TAO_LMVM and TAO_BLMVM.");
  ierr = MatLMVMGetJ0(M, H0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
  TaoLMVMGetH0KSP - Get the iterative solver for applying the inverse of the QN initial Hessian

  Input Parameters:
.  tao  - the Tao solver context

  Output Parameters:
.  ksp - KSP solver context for the initial Hessian

  Level: advanced

.seealso: TaoLMVMGetH0(), TaoLMVMGetH0KSP()
@*/
PetscErrorCode TaoLMVMGetH0KSP(Tao tao, KSP *ksp)
{
  TAO_LMVM       *lmP;
  TAO_BLMVM      *blmP;
  TaoType        type;
  PetscBool      is_lmvm, is_blmvm;
  Mat            M;
  PetscErrorCode ierr;

  ierr = TaoGetType(tao, &type);CHKERRQ(ierr);
  ierr = PetscStrcmp(type, TAOLMVM,  &is_lmvm);CHKERRQ(ierr);
  ierr = PetscStrcmp(type, TAOBLMVM, &is_blmvm);CHKERRQ(ierr);

  if (is_lmvm) {
    lmP = (TAO_LMVM *)tao->data;
    M = lmP->M;
  } else if (is_blmvm) {
    blmP = (TAO_BLMVM *)tao->data;
    M = blmP->M;
  } else SETERRQ(PetscObjectComm((PetscObject)tao), PETSC_ERR_ARG_WRONGSTATE, "This routine applies to TAO_LMVM and TAO_BLMVM.");
  ierr = MatLMVMGetJ0KSP(M, ksp);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

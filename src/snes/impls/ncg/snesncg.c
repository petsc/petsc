#include <../src/snes/impls/ncg/snesncgimpl.h>

#undef __FUNCT__
#define __FUNCT__ "SNESReset_NCG"
PetscErrorCode SNESReset_NCG(SNES snes)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (snes->work) {ierr = VecDestroyVecs(snes->nwork,&snes->work);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

/*
  SNESDestroy_NCG - Destroys the private SNES_NCG context that was created with SNESCreate_NCG().

  Input Parameter:
. snes - the SNES context

  Application Interface Routine: SNESDestroy()
*/
#undef __FUNCT__
#define __FUNCT__ "SNESDestroy_NCG"
PetscErrorCode SNESDestroy_NCG(SNES snes)
{
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  ierr = SNESReset_NCG(snes);CHKERRQ(ierr);
  ierr = PetscFree(snes->data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
   SNESSetUp_NCG - Sets up the internal data structures for the later use
   of the SNESNCG nonlinear solver.

   Input Parameters:
+  snes - the SNES context
-  x - the solution vector

   Application Interface Routine: SNESSetUp()
 */
#undef __FUNCT__
#define __FUNCT__ "SNESSetUp_NCG"
PetscErrorCode SNESSetUp_NCG(SNES snes)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = SNESDefaultGetWork(snes,4);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
/*
  SNESSetFromOptions_NCG - Sets various parameters for the SNESLS method.

  Input Parameter:
. snes - the SNES context

  Application Interface Routine: SNESSetFromOptions()
*/
#undef __FUNCT__
#define __FUNCT__ "SNESSetFromOptions_NCG"
static PetscErrorCode SNESSetFromOptions_NCG(SNES snes)
{
  SNES_NCG           *ncg     = (SNES_NCG *)snes->data;
  const char         *betas[] = {"FR","PRP","HS", "DY", "CD"};
  PetscErrorCode     ierr;
  PetscBool          debug, flg;
  PetscInt           indx;

  PetscFunctionBegin;
  ierr = PetscOptionsHead("SNES NCG options");CHKERRQ(ierr);
  ierr = PetscOptionsBool("-snes_ncg_monitor", "Monitor NCG iterations", "SNES", ncg->monitor ? PETSC_TRUE: PETSC_FALSE, &debug, PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEList("-snes_ncg_type","Nonlinear CG update used","", betas, 5, "FR",&indx, &flg);CHKERRQ(ierr);
  if (flg) {
    ncg->betatype = indx;
  }
  if (debug) {
    ncg->monitor = PETSC_VIEWER_STDOUT_(((PetscObject)snes)->comm);CHKERRQ(ierr);
  }
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
  SNESView_NCG - Prints info from the SNESNCG data structure.

  Input Parameters:
+ SNES - the SNES context
- viewer - visualization context

  Application Interface Routine: SNESView()
*/
#undef __FUNCT__
#define __FUNCT__ "SNESView_NCG"
static PetscErrorCode SNESView_NCG(SNES snes, PetscViewer viewer)
{
  PetscBool        iascii;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  ierr = PetscTypeCompare((PetscObject) viewer, PETSCVIEWERASCII, &iascii);CHKERRQ(ierr);
  if (iascii) {
    ierr = PetscViewerASCIIPrintf(viewer,"  line search type variant: %s\n", SNESLineSearchTypeName(snes->ls_type));CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESLineSearchExactLinear_NCG"
PetscErrorCode SNESLineSearchExactLinear_NCG(SNES snes,void *lsctx,Vec X,Vec F,Vec Y,PetscReal fnorm,PetscReal dummyXnorm,Vec G,Vec W,PetscReal *dummyYnorm,PetscReal *gnorm,PetscBool *flag)
{
  PetscScalar      alpha, ptAp;
  PetscErrorCode   ierr;
  /* SNES_NCG *ncg =  (SNES_NCG *) snes->data; */
  MatStructure     flg = DIFFERENT_NONZERO_PATTERN;

  PetscFunctionBegin;
  /*

   The exact step size for unpreconditioned linear CG is just:

   alpha = (r, r) / (p, Ap) = (f, f) / (y, Jy)

   */

  ierr = SNESComputeJacobian(snes, X, &snes->jacobian, &snes->jacobian_pre, &flg);CHKERRQ(ierr);
  ierr = VecDot(F, F, &alpha);CHKERRQ(ierr);
  ierr = MatMult(snes->jacobian, Y, W);CHKERRQ(ierr);
  ierr = VecDot(Y, W, &ptAp);CHKERRQ(ierr);
  alpha = alpha / ptAp;
  ierr = PetscPrintf(((PetscObject)snes)->comm, "alpha: %G\n", PetscRealPart(alpha));CHKERRQ(ierr);
  ierr = VecCopy(X, W);CHKERRQ(ierr);
  ierr = VecAXPY(W, alpha, Y);CHKERRQ(ierr);
  ierr = SNESComputeFunction(snes, W, G);CHKERRQ(ierr);
  ierr = VecNorm(G, NORM_2, gnorm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "SNESLineSearchSetType_NCG"
PetscErrorCode  SNESLineSearchSetType_NCG(SNES snes, SNESLineSearchType type)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;

  switch (type) {
  case SNES_LS_BASIC:
    ierr = SNESLineSearchSet(snes,SNESLineSearchNo,PETSC_NULL);CHKERRQ(ierr);
    break;
  case SNES_LS_BASIC_NONORMS:
    ierr = SNESLineSearchSet(snes,SNESLineSearchNoNorms,PETSC_NULL);CHKERRQ(ierr);
    break;
  case SNES_LS_QUADRATIC:
    ierr = SNESLineSearchSet(snes,SNESLineSearchQuadraticSecant,PETSC_NULL);CHKERRQ(ierr);
    break;
  case SNES_LS_TEST:
    ierr = SNESLineSearchSet(snes,SNESLineSearchExactLinear_NCG,PETSC_NULL);CHKERRQ(ierr);
    break;
  case SNES_LS_SECANT:
    ierr = SNESLineSearchSet(snes,SNESLineSearchSecant,PETSC_NULL);CHKERRQ(ierr);
    break;
  default:
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP,"Unknown line search type");
    break;
  }
  snes->ls_type = type;
  PetscFunctionReturn(0);
}
EXTERN_C_END

/*
  SNESSolve_NCG - Solves a nonlinear system with the Nonlinear Conjugate Gradient method.

  Input Parameters:
. snes - the SNES context

  Output Parameter:
. outits - number of iterations until termination

  Application Interface Routine: SNESSolve()
*/
#undef __FUNCT__
#define __FUNCT__ "SNESSolve_NCG"
PetscErrorCode SNESSolve_NCG(SNES snes)
{
  SNES_NCG            *ncg = (SNES_NCG *)snes->data;
  Vec                 X, dX, lX, F, W, B, G, Fold;
  PetscReal           fnorm, gnorm, dummyNorm, beta = 0.0;
  PetscScalar         dXdotF, dXolddotFold, dXdotFold, lXdotF, lXdotFold;
  PetscInt            maxits, i;
  PetscErrorCode      ierr;
  SNESConvergedReason reason;
  PetscBool           lsSuccess = PETSC_TRUE;
  PetscFunctionBegin;
  snes->reason = SNES_CONVERGED_ITERATING;

  maxits = snes->max_its;            /* maximum number of iterations */
  X      = snes->vec_sol;            /* X^n */
  Fold   = snes->work[0];            /* The previous iterate of X */
  dX     = snes->work[1];            /* the steepest direction */
  lX     = snes->vec_sol_update;     /* the conjugate direction */
  F      = snes->vec_func;           /* residual vector */
  W      = snes->work[2];            /* work vector and solution output for the line search */
  B      = snes->vec_rhs;            /* the right hand side */
  G      = snes->work[3];            /* the line search result function evaluation */

  ierr = PetscObjectTakeAccess(snes);CHKERRQ(ierr);
  snes->iter = 0;
  snes->norm = 0.;
  ierr = PetscObjectGrantAccess(snes);CHKERRQ(ierr);

  /* compute the initial function and preconditioned update delX */
  ierr = SNESComputeFunction(snes,X,F);CHKERRQ(ierr);
  if (snes->domainerror) {
    snes->reason = SNES_DIVERGED_FUNCTION_DOMAIN;
    PetscFunctionReturn(0);
  }
  /* convergence test */
  ierr = VecNorm(F, NORM_2, &fnorm);CHKERRQ(ierr); /* fnorm <- ||F||  */
  if (PetscIsInfOrNanReal(fnorm)) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_FP,"Infinite or not-a-number generated in norm");
  ierr = PetscObjectTakeAccess(snes);CHKERRQ(ierr);
  snes->norm = fnorm;
  ierr = PetscObjectGrantAccess(snes);CHKERRQ(ierr);
  SNESLogConvHistory(snes,fnorm,0);
  ierr = SNESMonitor(snes,0,fnorm);CHKERRQ(ierr);

  /* set parameter for default relative tolerance convergence test */
  snes->ttol = fnorm*snes->rtol;
  /* test convergence */
  ierr = (*snes->ops->converged)(snes,0,0.0,0.0,fnorm,&snes->reason,snes->cnvP);CHKERRQ(ierr);
  if (snes->reason) PetscFunctionReturn(0);

  /* Call general purpose update function */
  if (snes->ops->update) {
    ierr = (*snes->ops->update)(snes, snes->iter);CHKERRQ(ierr);
  }

  /* first update -- just use the (preconditioned) residual direction for the initial conjugate direction */
  if (!snes->pc) {
    ierr = VecCopy(F, lX);CHKERRQ(ierr);
    ierr = VecScale(lX, -1.0);CHKERRQ(ierr);
  } else {
    ierr = VecCopy(X, lX);CHKERRQ(ierr);
    ierr = SNESSolve(snes->pc, B, lX);CHKERRQ(ierr);
    ierr = SNESGetConvergedReason(snes->pc,&reason);CHKERRQ(ierr);
    ierr = VecAXPY(lX, -1.0, X);CHKERRQ(ierr);
    if (reason < 0 && (reason != SNES_DIVERGED_MAX_IT)) {
      snes->reason = SNES_DIVERGED_INNER;
      PetscFunctionReturn(0);
    }
  }
  ierr = VecDot(lX, F, &dXdotF);CHKERRQ(ierr);
  for(i = 1; i < maxits; i++) {
    lsSuccess = PETSC_TRUE;
    /* some update types require the old residual */
    ierr = VecCopy(F, Fold);CHKERRQ(ierr);
    ierr = (*snes->ops->linesearch)(snes, snes->lsP, X, F, lX, fnorm, 0.0, G, W, &dummyNorm, &gnorm, &lsSuccess);CHKERRQ(ierr);
    if (!lsSuccess) {
      if (++snes->numFailures >= snes->maxFailures) {
        snes->reason = SNES_DIVERGED_LINE_SEARCH;
        break;
      }
    }
    if (snes->nfuncs >= snes->max_funcs) {
      snes->reason = SNES_DIVERGED_FUNCTION_COUNT;
      break;
    }
    if (snes->domainerror) {
      snes->reason = SNES_DIVERGED_FUNCTION_DOMAIN;
      PetscFunctionReturn(0);
    }

    /* copy over the solution */
    ierr = VecCopy(G, F);CHKERRQ(ierr);
    ierr = VecCopy(W, X);CHKERRQ(ierr);
    fnorm = gnorm;

    /* Monitor convergence */
    ierr = PetscObjectTakeAccess(snes);CHKERRQ(ierr);
    snes->iter = i;
    snes->norm = fnorm;
    ierr = PetscObjectGrantAccess(snes);CHKERRQ(ierr);
    SNESLogConvHistory(snes,snes->norm,0);
    ierr = SNESMonitor(snes,snes->iter,snes->norm);CHKERRQ(ierr);

    /* Test for convergence */
    ierr = (*snes->ops->converged)(snes,snes->iter,0.0,0.0,fnorm,&snes->reason,snes->cnvP);CHKERRQ(ierr);
    if (snes->reason) break;

    /* Call general purpose update function */
    if (snes->ops->update) {
      ierr = (*snes->ops->update)(snes, snes->iter);CHKERRQ(ierr);
    }
    if (snes->usegs && snes->ops->computegs) {
      ierr = VecCopy(X, dX);CHKERRQ(ierr);
      ierr = SNESComputeGS(snes, B, dX);CHKERRQ(ierr);
      ierr = VecAXPY(dX, -1.0, X);CHKERRQ(ierr);
    } else if (snes->pc) {
      ierr = VecCopy(X,dX);CHKERRQ(ierr);
      ierr = SNESSolve(snes->pc, B, dX);CHKERRQ(ierr);
      ierr = SNESGetConvergedReason(snes->pc,&reason);CHKERRQ(ierr);
      if (reason < 0 && (reason != SNES_DIVERGED_MAX_IT)) {
        snes->reason = SNES_DIVERGED_INNER;
        PetscFunctionReturn(0);
      }
      ierr = VecAXPY(dX,-1.0,X);CHKERRQ(ierr);
    } else {
      ierr = VecCopy(F,dX);CHKERRQ(ierr);
      ierr = VecScale(dX,-1.0);CHKERRQ(ierr);
    }

    /* compute the conjugate direction lX = dX + beta*lX with beta = ((F, dX) / (F_old, dX_old) (Fletcher-Reeves update)*/
    switch(ncg->betatype) {
    case 0: /* Fletcher-Reeves */
      dXolddotFold = dXdotF;
      ierr = VecDot(dX, F, &dXdotF);CHKERRQ(ierr);
      beta = PetscRealPart(dXdotF / dXolddotFold);
      break;
    case 1: /* Polak-Ribiere-Poylak */
      dXolddotFold = dXdotF;
      ierr = VecDot(dX, Fold, &dXdotFold);CHKERRQ(ierr);
      ierr = VecDot(dX, F, &dXdotF);CHKERRQ(ierr);
      beta = PetscRealPart(((dXdotF - dXdotFold) / dXolddotFold));
      if (beta < 0.0) beta = 0.0; /* restart */
      break;
    case 2: /* Hestenes-Stiefel */
      ierr = VecDot(dX, F, &dXdotF);CHKERRQ(ierr);
      ierr = VecDot(dX, Fold, &dXdotFold);CHKERRQ(ierr);
      ierr = VecDot(lX, F, &lXdotF);CHKERRQ(ierr);
      ierr = VecDot(lX, Fold, &lXdotFold);CHKERRQ(ierr);
      beta = PetscRealPart((dXdotF - dXdotFold) / (lXdotF - lXdotFold));
      break;
    case 3: /* Dai-Yuan */
      ierr = VecDot(dX, F, &dXdotF);CHKERRQ(ierr);
      ierr = VecDot(lX, F, &lXdotF);CHKERRQ(ierr);
      ierr = VecDot(lX, Fold, &lXdotFold);CHKERRQ(ierr);
      beta = PetscRealPart(dXdotF / (lXdotFold - lXdotF));CHKERRQ(ierr);
      break;
    case 4: /* Conjugate Descent */
      ierr = VecDot(dX, F, &dXdotF);CHKERRQ(ierr);
      ierr = VecDot(lX, Fold, &lXdotFold);CHKERRQ(ierr);
      beta = PetscRealPart(dXdotF / lXdotFold);CHKERRQ(ierr);
      break;
    }
    if (ncg->monitor) {
      ierr = PetscViewerASCIIPrintf(ncg->monitor, "beta = %e\n", beta);CHKERRQ(ierr);
    }
    ierr = VecAYPX(lX, beta, dX);CHKERRQ(ierr);
    /* line search for the proper contribution of lX to the solution */
  }
  if (i == maxits) {
    ierr = PetscInfo1(snes, "Maximum number of iterations has been reached: %D\n", maxits);CHKERRQ(ierr);
    if (!snes->reason) snes->reason = SNES_DIVERGED_MAX_IT;
  }
  PetscFunctionReturn(0);
}

/*MC
  SNESNCG - Nonlinear Conjugate-Gradient method for the solution of general nonlinear systems.

  Level: beginner

  Options Database:
+   -snes_ls_damping - damping factor to apply to F(x) (used only if -snes_ls is basic or basicnonorms)
-   -snes_ls <basic,basicnormnorms,quadratic>

Notes: This solves the nonlinear system of equations F(x) = 0 using the nonlinear generalization of the conjugate
gradient method.  This may be used with a nonlinear preconditioner used to pick the new search directions, but otherwise
chooses the initial search direction as F(x) for the initial guess x.


.seealso:  SNESCreate(), SNES, SNESSetType(), SNESLS, SNESTR, SNESNGMRES, SNESNQN
M*/
EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "SNESCreate_NCG"
PetscErrorCode  SNESCreate_NCG(SNES snes)
{
  PetscErrorCode   ierr;
  SNES_NCG * neP;

  PetscFunctionBegin;
  snes->ops->destroy         = SNESDestroy_NCG;
  snes->ops->setup           = SNESSetUp_NCG;
  snes->ops->setfromoptions  = SNESSetFromOptions_NCG;
  snes->ops->view            = SNESView_NCG;
  snes->ops->solve           = SNESSolve_NCG;
  snes->ops->reset           = SNESReset_NCG;

  snes->usesksp              = PETSC_FALSE;
  snes->usespc               = PETSC_TRUE;

  ierr = PetscNewLog(snes, SNES_NCG, &neP);CHKERRQ(ierr);
  snes->data = (void*) neP;
  neP->monitor = PETSC_NULL;
  neP->betatype = 0;
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)snes,"SNESLineSearchSetType_C","SNESLineSearchSetType_NCG",SNESLineSearchSetType_NCG);CHKERRQ(ierr);
  ierr = SNESLineSearchSetType(snes, SNES_LS_QUADRATIC);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END

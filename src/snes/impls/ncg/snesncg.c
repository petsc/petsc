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
  SNES_NCG *neP = (SNES_NCG*)snes->data;

  PetscFunctionBegin;
  ierr = SNESReset_NCG(snes);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&neP->monitor);CHKERRQ(ierr);
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
  ierr = SNESDefaultGetWork(snes,2);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "SNESLineSearchSetMonitor_NCG"
PetscErrorCode  SNESLineSearchSetMonitor_NCG(SNES snes,PetscBool  flg)
{
  SNES_NCG *neP = (SNES_NCG*)snes->data;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  if (flg && !neP->monitor) {
    ierr = PetscViewerASCIIOpen(((PetscObject)snes)->comm,"stdout",&neP->monitor);CHKERRQ(ierr);
  } else if (!flg && neP->monitor) {
    ierr = PetscViewerDestroy(&neP->monitor);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}
EXTERN_C_END

PetscErrorCode SNESLineSearchNoNorms_NCG(SNES,void*,Vec,Vec,Vec,PetscReal,PetscReal,Vec,Vec,PetscReal*,PetscReal*,PetscBool*);
PetscErrorCode SNESLineSearchNo_NCG(SNES,void*,Vec,Vec,Vec,PetscReal,PetscReal,Vec,Vec,PetscReal*,PetscReal*,PetscBool*);
PetscErrorCode SNESLineSearchQuadratic_NCG(SNES,void*,Vec,Vec,Vec,PetscReal,PetscReal,Vec,Vec,PetscReal*,PetscReal*,PetscBool*);
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
  SNES_NCG   *ls = (SNES_NCG *)snes->data;
  SNESLineSearchType indx;
  PetscBool          flg,set;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  ierr = PetscOptionsHead("SNES NCG options");CHKERRQ(ierr);
    ierr = PetscOptionsEnum("-snes_ls","NCG line search type","SNESLineSearchSet",SNESLineSearchTypes,(PetscEnum)ls->type,(PetscEnum*)&indx,&flg);CHKERRQ(ierr);
    if (flg) {
      ls->type = indx;
      switch (indx) {
      case SNES_LS_BASIC:
        ierr = SNESLineSearchSet(snes,SNESLineSearchNo_NCG,PETSC_NULL);CHKERRQ(ierr);
        break;
      case SNES_LS_BASIC_NONORMS:
        ierr = SNESLineSearchSet(snes,SNESLineSearchNoNorms_NCG,PETSC_NULL);CHKERRQ(ierr);
        break;
      case SNES_LS_QUADRATIC:
        ierr = SNESLineSearchSet(snes,SNESLineSearchQuadratic_NCG,PETSC_NULL);CHKERRQ(ierr);
        break;
      case SNES_LS_CUBIC:
        SETERRQ(((PetscObject)snes)->comm,PETSC_ERR_SUP,"No support for cubic search");
        break;
      }
    }
    ierr = PetscOptionsReal("-snes_ls_damping","Damping parameter","SNES",ls->damping,&ls->damping,&flg);CHKERRQ(ierr);
    ierr = PetscOptionsBool("-snes_ls_monitor","Print progress of line searches","SNESLineSearchSetMonitor",ls->monitor ? PETSC_TRUE : PETSC_FALSE,&flg,&set);CHKERRQ(ierr);
    if (set) {ierr = SNESLineSearchSetMonitor(snes,flg);CHKERRQ(ierr);}
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
  SNES_NCG *ls = (SNES_NCG *)snes->data;
  PetscBool        iascii;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  ierr = PetscTypeCompare((PetscObject) viewer, PETSCVIEWERASCII, &iascii);CHKERRQ(ierr);
  if (iascii) {
    ierr = PetscViewerASCIIPrintf(viewer,"  line search type variant: %s\n", SNESLineSearchTypeName(ls->type));CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESLineSearchNo_NCG"
PetscErrorCode SNESLineSearchNo_NCG(SNES snes,void *lsctx,Vec X,Vec F,Vec Y,PetscReal fnorm,PetscReal dummyXnorm,Vec dummyG,Vec W,PetscReal *dummyYnorm,PetscReal *gnorm,PetscBool *flag)
{
  PetscErrorCode   ierr;
  SNES_NCG *neP = (SNES_NCG *) snes->data;

  PetscFunctionBegin;
  ierr = VecAXPY(X, neP->damping, Y);CHKERRQ(ierr);
  ierr = SNESComputeFunction(snes, X, F);CHKERRQ(ierr);
  ierr = VecNorm(F, NORM_2, gnorm);CHKERRQ(ierr);
  if (PetscIsInfOrNanReal(*gnorm)) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_FP,"Infinite or not-a-number generated norm");
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESLineSearchNoNorms_NCG"
PetscErrorCode SNESLineSearchNoNorms_NCG(SNES snes,void *lsctx,Vec X,Vec F,Vec Y,PetscReal fnorm,PetscReal dummyXnorm,Vec dummyG,Vec W,PetscReal *dummyYnorm,PetscReal *gnorm,PetscBool *flag)
{
  PetscErrorCode   ierr;
  SNES_NCG *neP = (SNES_NCG *) snes->data;

  PetscFunctionBegin;
  ierr = VecAXPY(X, neP->damping, Y);CHKERRQ(ierr);
  ierr = SNESComputeFunction(snes, X, F);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESLineSearchQuadratic_NCG"
PetscErrorCode SNESLineSearchQuadratic_NCG(SNES snes,void *lsctx,Vec X,Vec F,Vec Y,PetscReal fnorm,PetscReal dummyXnorm,Vec G,Vec W,PetscReal *dummyYnorm,PetscReal *gnorm,PetscBool *flag)
{
  PetscInt         i;
  PetscReal        alphas[3] = {0.0, 0.5, 1.0};
  PetscReal        norms[3];
  PetscReal        alpha,a,b;
  PetscErrorCode   ierr;
  SNES_NCG *neP = (SNES_NCG *) snes->data;

  PetscFunctionBegin;
  norms[0]  = fnorm;
  for(i=1; i < 3; ++i) {
    ierr = VecWAXPY(W, alphas[i], Y, X);CHKERRQ(ierr);     /* W =  X^n - \alpha Y */
    ierr = SNESComputeFunction(snes, W, F);CHKERRQ(ierr);
    ierr = VecNorm(F, NORM_2, &norms[i]);CHKERRQ(ierr);
  }
  for(i = 0; i < 3; ++i) {
    norms[i] = PetscSqr(norms[i]);
  }
  /* Fit a quadratic:
       If we have x_{0,1,2} = 0, x_1, x_2 which generate norms y_{0,1,2}
       a = (x_1 y_2 - x_2 y_1 + (x_2 - x_1) y_0)/(x^2_2 x_1 - x_2 x^2_1)
       b = (x^2_1 y_2 - x^2_2 y_1 + (x^2_2 - x^2_1) y_0)/(x_2 x^2_1 - x^2_2 x_1)
       c = y_0
       x_min = -b/2a

       If we let x_{0,1,2} = 0, 0.5, 1.0
       a = 2 y_2 - 4 y_1 + 2 y_0
       b =  -y_2 + 4 y_1 - 3 y_0
       c =   y_0
  */
  a = (alphas[1]*norms[2] - alphas[2]*norms[1] + (alphas[2] - alphas[1])*norms[0])/(PetscSqr(alphas[2])*alphas[1] - alphas[2]*PetscSqr(alphas[1]));
  b = (PetscSqr(alphas[1])*norms[2] - PetscSqr(alphas[2])*norms[1] + (PetscSqr(alphas[2]) - PetscSqr(alphas[1]))*norms[0])/(alphas[2]*PetscSqr(alphas[1]) - PetscSqr(alphas[2])*alphas[1]);
  /* Check for positive a (concave up) */
  if (a >= 0.0) {
    alpha = -b/(2.0*a);
    alpha = PetscMin(alpha, alphas[2]);
    alpha = PetscMax(alpha, alphas[0]);
  } else {
    alpha = 1.0;
  }
  if (neP->monitor) {
    ierr = PetscViewerASCIIAddTab(neP->monitor,((PetscObject)snes)->tablevel);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(neP->monitor,"    Line search: norms[0] = %g, norms[1] = %g, norms[2] = %g alpha %g\n", sqrt(norms[0]),sqrt(norms[1]),sqrt(norms[2]),alpha);CHKERRQ(ierr);
    ierr = PetscViewerASCIISubtractTab(neP->monitor,((PetscObject)snes)->tablevel);CHKERRQ(ierr);
  }
  ierr = VecAXPY(X, alpha, Y);CHKERRQ(ierr);
  if (alpha != 1.0) {
    ierr = SNESComputeFunction(snes, X, F);CHKERRQ(ierr);
    ierr = VecNorm(F, NORM_2, gnorm);CHKERRQ(ierr);
  } else {
    *gnorm = PetscSqrtReal(norms[2]); 
  }
  if (alpha == 0.0) *flag = PETSC_FALSE;
  else              *flag = PETSC_TRUE;
  PetscFunctionReturn(0);
}

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
  SNES_NCG            *neP = (SNES_NCG *) snes->data;
  Vec                 X, dX, lX, F, W;
  PetscReal           fnorm, dummyNorm;
  PetscScalar         dXdot, dXdot_old;
  PetscInt            maxits, i;
  PetscErrorCode      ierr;
  SNESConvergedReason reason;
  PetscBool           lsSuccess = PETSC_TRUE;
  PetscFunctionBegin;
  snes->reason = SNES_CONVERGED_ITERATING;

  maxits = snes->max_its;            /* maximum number of iterations */
  X      = snes->vec_sol;            /* X^n */
  dX     = snes->vec_sol_update;     /* the steepest direction */
  lX     = snes->work[1];            /* the conjugate direction */
  F      = snes->vec_func;           /* residual vector */
  W      = snes->work[0];            /* work vector for the line search */

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
    ierr = SNESSolve(snes->pc, 0, lX);CHKERRQ(ierr);
    ierr = SNESGetConvergedReason(snes->pc,&reason);CHKERRQ(ierr);
    ierr = VecAXPY(dX, -1.0, X);CHKERRQ(ierr);
    if (reason < 0 && reason != SNES_DIVERGED_MAX_IT) {
      snes->reason = SNES_DIVERGED_INNER;
      PetscFunctionReturn(0);
    }
  }
  ierr = VecDot(lX, lX, &dXdot);CHKERRQ(ierr);
  /* line search for the update */
  ierr = (*neP->LineSearch)(snes, neP->lsP, X, F, dX, fnorm, 0.0, 0, W, &dummyNorm, &fnorm, &lsSuccess);CHKERRQ(ierr);

  for(i = 1; i < maxits; i++) {
    lsSuccess = PETSC_TRUE;
    /* Call general purpose update function */
    if (snes->ops->update) {
      ierr = (*snes->ops->update)(snes, snes->iter);CHKERRQ(ierr);
    }
    if (!snes->pc) {
      ierr = VecCopy(F,dX);CHKERRQ(ierr);
      ierr = VecScale(dX,-1.0);CHKERRQ(ierr);
    } else {
      ierr = VecCopy(X,dX);CHKERRQ(ierr);
      ierr = SNESSolve(snes->pc, 0, dX);CHKERRQ(ierr);
      ierr = SNESGetConvergedReason(snes->pc,&reason);CHKERRQ(ierr);
      if (reason < 0) {
        snes->reason = SNES_DIVERGED_INNER;
        PetscFunctionReturn(0);
      }
      ierr = VecAXPY(dX,-1.0,X);CHKERRQ(ierr);
    }

    /* compute the conjugate direction lX = dX + beta*lX with beta = (dX, dX) / (DX_old, dX_old) (Fletcher-Reeves Update)*/
    dXdot_old = dXdot;
    ierr = VecDot(dX, dX, &dXdot);CHKERRQ(ierr);
    ierr = VecAXPY(lX, dXdot / dXdot_old, dX);CHKERRQ(ierr);
    /* line search for the proper contribution of lX to the solution */
    ierr = (*neP->LineSearch)(snes, neP->lsP, X, F, lX, fnorm, 0.0, 0, W, &dummyNorm, &fnorm, &lsSuccess);CHKERRQ(ierr);
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
    /* Monitor convergence */
    ierr = PetscObjectTakeAccess(snes);CHKERRQ(ierr);
    snes->iter = i+1;
    snes->norm = fnorm;
    ierr = PetscObjectGrantAccess(snes);CHKERRQ(ierr);
    SNESLogConvHistory(snes,snes->norm,0);
    ierr = SNESMonitor(snes,snes->iter,snes->norm);CHKERRQ(ierr);
    /* Test for convergence */
    ierr = (*snes->ops->converged)(snes,snes->iter,0.0,0.0,fnorm,&snes->reason,snes->cnvP);CHKERRQ(ierr);
    if (snes->reason) break;
  }
  if (i == maxits) {
    ierr = PetscInfo1(snes, "Maximum number of iterations has been reached: %D\n", maxits);CHKERRQ(ierr);
    if (!snes->reason) snes->reason = SNES_DIVERGED_MAX_IT;
  }
  PetscFunctionReturn(0);
}

typedef PetscErrorCode (*FCN1)(SNES,Vec,Vec,void*,PetscBool *);                 /* force argument to next function to not be extern C*/
EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "SNESLineSearchSetPreCheck_NCG"
PetscErrorCode  SNESLineSearchSetPreCheck_NCG(SNES snes, FCN1 func, void *checkctx)
{
  PetscFunctionBegin;
  ((SNES_NCG *)(snes->data))->precheckstep = func;
  ((SNES_NCG *)(snes->data))->precheck     = checkctx;
  PetscFunctionReturn(0);
}
EXTERN_C_END

typedef PetscErrorCode (*FCN2)(SNES,void*,Vec,Vec,Vec,PetscReal,PetscReal,Vec,Vec,PetscReal*,PetscReal*,PetscBool *); /* force argument to next function to not be extern C*/
EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "SNESLineSearchSet_NCG"
PetscErrorCode  SNESLineSearchSet_NCG(SNES snes, FCN2 func, void *lsctx)
{
  PetscFunctionBegin;
  ((SNES_NCG *)(snes->data))->LineSearch = func;
  ((SNES_NCG *)(snes->data))->lsP        = lsctx;
  PetscFunctionReturn(0);
}
EXTERN_C_END

typedef PetscErrorCode (*FCN3)(SNES,Vec,Vec,Vec,void*,PetscBool *,PetscBool *); /* force argument to next function to not be extern C*/
EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "SNESLineSearchSetPostCheck_NCG"
PetscErrorCode  SNESLineSearchSetPostCheck_NCG(SNES snes, FCN3 func, void *checkctx)
{
  PetscFunctionBegin;
  ((SNES_NCG *)(snes->data))->postcheckstep = func;
  ((SNES_NCG *)(snes->data))->postcheck     = checkctx;
  PetscFunctionReturn(0);
}
EXTERN_C_END

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
  SNES_NCG *neP;
  PetscErrorCode   ierr;

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
  neP->maxstep       = 1.e8;
  neP->steptol       = 1.e-12;
  neP->type          = SNES_LS_QUADRATIC;
  neP->damping       = 1.0;
  neP->LineSearch    = SNESLineSearchQuadratic_NCG;
  neP->lsP           = PETSC_NULL;
  neP->postcheckstep = PETSC_NULL;
  neP->postcheck     = PETSC_NULL;
  neP->precheckstep  = PETSC_NULL;
  neP->precheck      = PETSC_NULL;

  ierr = PetscObjectComposeFunctionDynamic((PetscObject)snes,"SNESLineSearchSetMonitor_C","SNESLineSearchSetMonitor_NCG",SNESLineSearchSetMonitor_NCG);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)snes,"SNESLineSearchSet_C","SNESLineSearchSet_NCG",SNESLineSearchSet_NCG);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)snes,"SNESLineSearchSetPostCheck_C","SNESLineSearchSetPostCheck_NCG",SNESLineSearchSetPostCheck_NCG);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)snes,"SNESLineSearchSetPreCheck_C","SNESLineSearchSetPreCheck_NCG",SNESLineSearchSetPreCheck_NCG);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END

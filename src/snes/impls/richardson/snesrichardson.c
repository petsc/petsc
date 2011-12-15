
#include <../src/snes/impls/richardson/snesrichardsonimpl.h>

#undef __FUNCT__
#define __FUNCT__ "SNESReset_NRichardson"
PetscErrorCode SNESReset_NRichardson(SNES snes)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (snes->work) {ierr = VecDestroyVecs(snes->nwork,&snes->work);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

/*
  SNESDestroy_NRichardson - Destroys the private SNES_NRichardson context that was created with SNESCreate_NRichardson().

  Input Parameter:
. snes - the SNES context

  Application Interface Routine: SNESDestroy()
*/
#undef __FUNCT__
#define __FUNCT__ "SNESDestroy_NRichardson"
PetscErrorCode SNESDestroy_NRichardson(SNES snes)
{
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  ierr = SNESReset_NRichardson(snes);CHKERRQ(ierr);
  ierr = PetscFree(snes->data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
   SNESSetUp_NRichardson - Sets up the internal data structures for the later use
   of the SNESNRICHARDSON nonlinear solver.

   Input Parameters:
+  snes - the SNES context
-  x - the solution vector

   Application Interface Routine: SNESSetUp()
 */
#undef __FUNCT__
#define __FUNCT__ "SNESSetUp_NRichardson"
PetscErrorCode SNESSetUp_NRichardson(SNES snes)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = SNESDefaultGetWork(snes,2);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
  SNESSetFromOptions_NRichardson - Sets various parameters for the SNESLS method.

  Input Parameter:
. snes - the SNES context

  Application Interface Routine: SNESSetFromOptions()
*/
#undef __FUNCT__
#define __FUNCT__ "SNESSetFromOptions_NRichardson"
static PetscErrorCode SNESSetFromOptions_NRichardson(SNES snes)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
    ierr = PetscOptionsHead("SNES Richardson options");CHKERRQ(ierr);
    ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
  SNESView_NRichardson - Prints info from the SNESRichardson data structure.

  Input Parameters:
+ SNES - the SNES context
- viewer - visualization context

  Application Interface Routine: SNESView()
*/
#undef __FUNCT__
#define __FUNCT__ "SNESView_NRichardson"
static PetscErrorCode SNESView_NRichardson(SNES snes, PetscViewer viewer)
{
  PetscBool        iascii;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  ierr = PetscTypeCompare((PetscObject) viewer, PETSCVIEWERASCII, &iascii);CHKERRQ(ierr);
  if (iascii) {
    ierr = PetscViewerASCIIPrintf(viewer,"  richardson variant: %s\n", SNESLineSearchTypeName(snes->ls_type));CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESLineSearchQuadratic_NRichardson"
PetscErrorCode SNESLineSearchQuadratic_NRichardson(SNES snes,void *lsctx,Vec X,Vec F,Vec Y,PetscReal fnorm,PetscReal dummyXnorm,Vec G,Vec W,PetscReal *dummyYnorm,PetscReal *gnorm,PetscBool *flag)
{
  PetscInt         i;
  PetscReal        alphas[3] = {0.0, 0.5, 1.0};
  PetscReal        norms[3];
  PetscReal        alpha,a,b;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  norms[0]  = fnorm;
  for(i=1; i < 3; ++i) {
    ierr = VecWAXPY(W, -alphas[i], Y, X);CHKERRQ(ierr);     /* W =  X^n - \alpha Y */
    ierr = SNESComputeFunction(snes, W, G);CHKERRQ(ierr);
    ierr = VecNorm(G, NORM_2, &norms[i]);CHKERRQ(ierr);
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
  if (snes->ls_monitor) {
    ierr = PetscViewerASCIIAddTab(snes->ls_monitor,((PetscObject)snes)->tablevel);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(snes->ls_monitor,"    Line search: norms[0] = %g, norms[1] = %g, norms[2] = %g alpha %g\n", 
                                  PetscSqrtReal(norms[0]),PetscSqrtReal(norms[1]),PetscSqrtReal(norms[2]),alpha);CHKERRQ(ierr);
    ierr = PetscViewerASCIISubtractTab(snes->ls_monitor,((PetscObject)snes)->tablevel);CHKERRQ(ierr);
  }
  ierr = VecCopy(X, W);CHKERRQ(ierr);
  ierr = VecAXPY(W, -alpha, Y);CHKERRQ(ierr);
  if (alpha != 1.0) {
    ierr = SNESComputeFunction(snes, W, G);CHKERRQ(ierr);
    ierr = VecNorm(G, NORM_2, gnorm);CHKERRQ(ierr);
  } else {
    *gnorm = PetscSqrtReal(norms[2]);
  }
  if (alpha == 0.0) *flag = PETSC_FALSE;
  else              *flag = PETSC_TRUE;
  PetscFunctionReturn(0);
}

/*
  SNESSolve_NRichardson - Solves a nonlinear system with the Richardson method.

  Input Parameters:
. snes - the SNES context

  Output Parameter:
. outits - number of iterations until termination

  Application Interface Routine: SNESSolve()
*/
#undef __FUNCT__
#define __FUNCT__ "SNESSolve_NRichardson"
PetscErrorCode SNESSolve_NRichardson(SNES snes)
{
  Vec                 X, Y, F, W, G;
  PetscReal           fnorm;
  PetscInt            maxits, i;
  PetscErrorCode      ierr;
  SNESConvergedReason reason;

  PetscFunctionBegin;
  snes->reason = SNES_CONVERGED_ITERATING;

  maxits = snes->max_its;        /* maximum number of iterations */
  X      = snes->vec_sol;        /* X^n */
  Y      = snes->vec_sol_update; /* \tilde X */
  F      = snes->vec_func;       /* residual vector */
  W      = snes->work[0];        /* work vector */
  G      = snes->work[1];        /* line search function vector */

  ierr = PetscObjectTakeAccess(snes);CHKERRQ(ierr);
  snes->iter = 0;
  snes->norm = 0.;
  ierr = PetscObjectGrantAccess(snes);CHKERRQ(ierr);
  ierr = SNESComputeFunction(snes,X,F);CHKERRQ(ierr);
  if (snes->domainerror) {
    snes->reason = SNES_DIVERGED_FUNCTION_DOMAIN;
    PetscFunctionReturn(0);
  }
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

  for(i = 0; i < maxits; i++) {
    PetscBool  lsSuccess = PETSC_TRUE;
    PetscReal  dummyNorm;

    /* Call general purpose update function */
    if (snes->ops->update) {
      ierr = (*snes->ops->update)(snes, snes->iter);CHKERRQ(ierr);
    }
    if (snes->usegs && snes->ops->computegs) {
      ierr = VecCopy(X, Y);CHKERRQ(ierr);
      ierr = SNESComputeGS(snes, snes->vec_rhs, Y);CHKERRQ(ierr);
      ierr = VecAXPY(Y, -1.0, X);CHKERRQ(ierr);
    } else if (snes->pc) {
      ierr = VecCopy(X,Y);CHKERRQ(ierr);
      ierr = SNESSolve(snes->pc, snes->vec_rhs, Y);CHKERRQ(ierr);
      ierr = SNESGetConvergedReason(snes->pc,&reason);CHKERRQ(ierr);
      if (reason < 0  && reason != SNES_DIVERGED_MAX_IT) {
        snes->reason = SNES_DIVERGED_INNER;
        PetscFunctionReturn(0);
      }
      ierr = VecAXPY(Y,-1.0,X);CHKERRQ(ierr);
    } else {
      ierr = VecCopy(F,Y);CHKERRQ(ierr);
    }
    ierr = (*snes->ops->linesearch)(snes, snes->lsP, X, F, Y, fnorm, 0.0, G, W, &dummyNorm, &fnorm, &lsSuccess);CHKERRQ(ierr);
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
    ierr = VecCopy(G, F);CHKERRQ(ierr);
    ierr = VecCopy(W, X);CHKERRQ(ierr);
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


EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "SNESLineSearchSetType_NRichardson"
PetscErrorCode  SNESLineSearchSetType_NRichardson(SNES snes, SNESLineSearchType type)
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
    ierr = SNESLineSearchSet(snes,SNESLineSearchQuadratic_NRichardson,PETSC_NULL);CHKERRQ(ierr);
    break;
  case SNES_LS_SECANT:
    ierr = SNESLineSearchSet(snes,SNESLineSearchQuadraticSecant,PETSC_NULL);CHKERRQ(ierr);
    break;
  default:
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP,"Unknown line search type");
    break;
  }
  snes->ls_type = type;
  PetscFunctionReturn(0);
}
EXTERN_C_END

/*MC
  SNESNRICHARDSON - Richardson nonlinear solver that uses successive substitutions, also sometimes known as Picard iteration.

  Level: beginner

  Options Database:
+   -snes_ls_damping - damping factor to apply to F(x) (used only if -snes_ls is basic or basicnonorms)
-   -snes_ls <basic,basicnormnorms,quadratic>

  Notes: If no inner nonlinear preconditioner is provided then solves F(x) - b = 0 using x^{n+1} = x^{n} - lambda
            (F(x^n) - b) where lambda is obtained either SNESLineSearchSetDamping(), -snes_damping or a line search.  If
            an inner nonlinear preconditioner is provided (either with -npc_snes_type or SNESSetPC()) then the inner
            solver is called an initial solution x^n and the nonlinear Richardson uses x^{n+1} = x^{n} + lambda d^{n}
            where d^{n} = \hat{x}^{n} - x^{n} where \hat{x}^{n} is the solution returned from the inner solver.

     This uses no derivative information thus will be much slower then Newton's method obtained with -snes_type ls

.seealso:  SNESCreate(), SNES, SNESSetType(), SNESLS, SNESTR, SNESNGMRES, SNESNQN
M*/
EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "SNESCreate_NRichardson"
PetscErrorCode  SNESCreate_NRichardson(SNES snes)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  snes->ops->destroy         = SNESDestroy_NRichardson;
  snes->ops->setup           = SNESSetUp_NRichardson;
  snes->ops->setfromoptions  = SNESSetFromOptions_NRichardson;
  snes->ops->view            = SNESView_NRichardson;
  snes->ops->solve           = SNESSolve_NRichardson;
  snes->ops->reset           = SNESReset_NRichardson;

  snes->usesksp              = PETSC_FALSE;
  snes->usespc               = PETSC_TRUE;

  ierr = PetscObjectComposeFunctionDynamic((PetscObject)snes,"SNESLineSearchSetType_C","SNESLineSearchSetType_NRichardson",SNESLineSearchSetType_NRichardson);CHKERRQ(ierr);
  ierr = SNESLineSearchSetType(snes, SNES_LS_SECANT);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}
EXTERN_C_END

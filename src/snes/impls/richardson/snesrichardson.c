
#include <../src/snes/impls/richardson/snesrichardsonimpl.h>

#undef __FUNCT__
#define __FUNCT__ "SNESReset_Richardson"
PetscErrorCode SNESReset_Richardson(SNES snes)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (snes->work) {ierr = VecDestroyVecs(snes->nwork,&snes->work);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

/*
  SNESDestroy_Richardson - Destroys the private SNES_Richardson context that was created with SNESCreate_Richardson().

  Input Parameter:
. snes - the SNES context

  Application Interface Routine: SNESDestroy()
*/
#undef __FUNCT__  
#define __FUNCT__ "SNESDestroy_Richardson"
PetscErrorCode SNESDestroy_Richardson(SNES snes)
{
  PetscErrorCode ierr;
  SNES_Richardson    *neP = (SNES_Richardson*)snes->data;

  PetscFunctionBegin;
  ierr = SNESReset_Richardson(snes);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&neP->monitor);CHKERRQ(ierr);
  ierr = PetscFree(snes->data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
   SNESSetUp_Richardson - Sets up the internal data structures for the later use
   of the SNESRICHARDSON nonlinear solver.

   Input Parameters:
+  snes - the SNES context
-  x - the solution vector

   Application Interface Routine: SNESSetUp()
 */
#undef __FUNCT__  
#define __FUNCT__ "SNESSetUp_Richardson"
PetscErrorCode SNESSetUp_Richardson(SNES snes)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = SNESDefaultGetWork(snes,1);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "SNESLineSearchSetMonitor_Richardson"
PetscErrorCode  SNESLineSearchSetMonitor_Richardson(SNES snes,PetscBool  flg)
{
  SNES_Richardson    *neP = (SNES_Richardson*)snes->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (flg && !neP->monitor) {
    ierr = PetscViewerASCIIOpen(((PetscObject)snes)->comm,"stdout",&neP->monitor);CHKERRQ(ierr);
  } else if (!flg && neP->monitor) {
    ierr = PetscViewerDestroy(&neP->monitor);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}
EXTERN_C_END

PetscErrorCode SNESLineSearchNoNorms_Richardson(SNES,void*,Vec,Vec,Vec,PetscReal,PetscReal,Vec,Vec,PetscReal*,PetscReal*,PetscBool*);
PetscErrorCode SNESLineSearchNo_Richardson(SNES,void*,Vec,Vec,Vec,PetscReal,PetscReal,Vec,Vec,PetscReal*,PetscReal*,PetscBool*);
PetscErrorCode SNESLineSearchQuadratic_Richardson(SNES,void*,Vec,Vec,Vec,PetscReal,PetscReal,Vec,Vec,PetscReal*,PetscReal*,PetscBool*);
/*
  SNESSetFromOptions_Richardson - Sets various parameters for the SNESLS method.

  Input Parameter:
. snes - the SNES context

  Application Interface Routine: SNESSetFromOptions()
*/
#undef __FUNCT__  
#define __FUNCT__ "SNESSetFromOptions_Richardson"
static PetscErrorCode SNESSetFromOptions_Richardson(SNES snes)
{
  SNES_Richardson        *ls = (SNES_Richardson *)snes->data;
  SNESLineSearchType indx;
  PetscBool          flg,set;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  ierr = PetscOptionsHead("SNES Richardson options");CHKERRQ(ierr);
    ierr = PetscOptionsEnum("-snes_ls","Richardson line search type","SNESLineSearchSet",SNESLineSearchTypes,(PetscEnum)ls->type,(PetscEnum*)&indx,&flg);CHKERRQ(ierr);
    if (flg) {
      ls->type = indx;
      switch (indx) {
      case SNES_LS_BASIC:
        ierr = SNESLineSearchSet(snes,SNESLineSearchNo_Richardson,PETSC_NULL);CHKERRQ(ierr);
        break;
      case SNES_LS_BASIC_NONORMS:
        ierr = SNESLineSearchSet(snes,SNESLineSearchNoNorms_Richardson,PETSC_NULL);CHKERRQ(ierr);
        break;
      case SNES_LS_QUADRATIC:
        ierr = SNESLineSearchSet(snes,SNESLineSearchQuadratic_Richardson,PETSC_NULL);CHKERRQ(ierr);
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
  SNESView_Richardson - Prints info from the SNESRichardson data structure.

  Input Parameters:
+ SNES - the SNES context
- viewer - visualization context

  Application Interface Routine: SNESView()
*/
#undef __FUNCT__  
#define __FUNCT__ "SNESView_Richardson"
static PetscErrorCode SNESView_Richardson(SNES snes, PetscViewer viewer)
{
  SNES_Richardson   *ls = (SNES_Richardson *)snes->data;
  PetscBool      iascii;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscTypeCompare((PetscObject) viewer, PETSCVIEWERASCII, &iascii);CHKERRQ(ierr);
  if (iascii) {
    ierr = PetscViewerASCIIPrintf(viewer,"  richardson variant: %s\n", SNESLineSearchTypeName(ls->type));CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESLineSearchNo_Richardson"
PetscErrorCode SNESLineSearchNo_Richardson(SNES snes,void *lsctx,Vec X,Vec F,Vec Y,PetscReal fnorm,PetscReal dummyXnorm,Vec dummyG,Vec W,PetscReal *dummyYnorm,PetscReal *gnorm,PetscBool *flag)
{
  PetscErrorCode ierr;
  SNES_Richardson    *neP = (SNES_Richardson *) snes->data;

  PetscFunctionBegin;
  ierr = VecAXPY(X, neP->damping, Y);CHKERRQ(ierr);
  ierr = SNESComputeFunction(snes, X, F);CHKERRQ(ierr);
  ierr = VecNorm(F, NORM_2, gnorm);CHKERRQ(ierr);
  if (PetscIsInfOrNanReal(*gnorm)) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_FP,"Infinite or not-a-number generated norm");
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESLineSearchNoNorms_Richardson"
PetscErrorCode SNESLineSearchNoNorms_Richardson(SNES snes,void *lsctx,Vec X,Vec F,Vec Y,PetscReal fnorm,PetscReal dummyXnorm,Vec dummyG,Vec W,PetscReal *dummyYnorm,PetscReal *gnorm,PetscBool *flag)
{
  PetscErrorCode ierr;
  SNES_Richardson    *neP = (SNES_Richardson *) snes->data;

  PetscFunctionBegin;
  ierr = VecAXPY(X, neP->damping, Y);CHKERRQ(ierr);
  ierr = SNESComputeFunction(snes, X, F);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESLineSearchQuadratic_Richardson"
PetscErrorCode SNESLineSearchQuadratic_Richardson(SNES snes,void *lsctx,Vec X,Vec F,Vec Y,PetscReal fnorm,PetscReal dummyXnorm,Vec G,Vec W,PetscReal *dummyYnorm,PetscReal *gnorm,PetscBool *flag)
{
  PetscInt       i;
  PetscReal      alphas[3] = {0.0, 0.5, 1.0};
  PetscReal      norms[3];
  PetscReal      alpha,a,b;
  PetscErrorCode ierr;
  SNES_Richardson    *neP = (SNES_Richardson *) snes->data;

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
  SNESSolve_Richardson - Solves a nonlinear system with the Richardson method.

  Input Parameters:
. snes - the SNES context

  Output Parameter:
. outits - number of iterations until termination

  Application Interface Routine: SNESSolve()
*/
#undef __FUNCT__  
#define __FUNCT__ "SNESSolve_Richardson"
PetscErrorCode SNESSolve_Richardson(SNES snes)
{ 
  SNES_Richardson         *neP = (SNES_Richardson *) snes->data;
  Vec                 X, Y, F, W;
  PetscReal           fnorm;
  PetscInt            maxits, i;
  PetscErrorCode      ierr;
  SNESConvergedReason reason;

  PetscFunctionBegin;
  snes->reason = SNES_CONVERGED_ITERATING;

  maxits = snes->max_its;	     /* maximum number of iterations */
  X      = snes->vec_sol;	     /* X^n */
  Y      = snes->vec_sol_update; /* \tilde X */
  F      = snes->vec_func;       /* residual vector */
  W      = snes->work[0];        /* work vector */

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
    if (!snes->pc) {
      ierr = VecCopy(F,Y);CHKERRQ(ierr);
      ierr = VecScale(Y,-1.0);CHKERRQ(ierr);
    } else {
      ierr = VecCopy(X,Y);CHKERRQ(ierr);
      ierr = SNESSolve(snes->pc, 0, Y);CHKERRQ(ierr);
      ierr = SNESGetConvergedReason(snes->pc,&reason);CHKERRQ(ierr);
      if (reason < 0) {
        snes->reason = SNES_DIVERGED_INNER;
        PetscFunctionReturn(0);
      }
      ierr = VecAXPY(Y,-1.0,X);CHKERRQ(ierr);
    }

      ierr = (*neP->LineSearch)(snes, neP->lsP, X, F, Y, fnorm, 0.0, 0, W, &dummyNorm, &fnorm, &lsSuccess);CHKERRQ(ierr);
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
#define __FUNCT__ "SNESLineSearchSetPreCheck_Richardson"
PetscErrorCode  SNESLineSearchSetPreCheck_Richardson(SNES snes, FCN1 func, void *checkctx)
{
  PetscFunctionBegin;
  ((SNES_Richardson *)(snes->data))->precheckstep = func;
  ((SNES_Richardson *)(snes->data))->precheck     = checkctx;
  PetscFunctionReturn(0);
}
EXTERN_C_END

typedef PetscErrorCode (*FCN2)(SNES,void*,Vec,Vec,Vec,PetscReal,PetscReal,Vec,Vec,PetscReal*,PetscReal*,PetscBool *); /* force argument to next function to not be extern C*/
EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "SNESLineSearchSet_Richardson"
PetscErrorCode  SNESLineSearchSet_Richardson(SNES snes, FCN2 func, void *lsctx)
{
  PetscFunctionBegin;
  ((SNES_Richardson *)(snes->data))->LineSearch = func;
  ((SNES_Richardson *)(snes->data))->lsP        = lsctx;
  PetscFunctionReturn(0);
}
EXTERN_C_END

typedef PetscErrorCode (*FCN3)(SNES,Vec,Vec,Vec,void*,PetscBool *,PetscBool *); /* force argument to next function to not be extern C*/
EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "SNESLineSearchSetPostCheck_Richardson"
PetscErrorCode  SNESLineSearchSetPostCheck_Richardson(SNES snes, FCN3 func, void *checkctx)
{
  PetscFunctionBegin;
  ((SNES_Richardson *)(snes->data))->postcheckstep = func;
  ((SNES_Richardson *)(snes->data))->postcheck     = checkctx;
  PetscFunctionReturn(0);
}
EXTERN_C_END

/*MC
  SNESRICHARDSON - Richardson nonlinear solver that uses successive substitutions

  Level: beginner

  Options Database:
+   -snes_ls_damping - damping factor to apply to F(x) (used only if -snes_ls is basic or basicnonorms)
-   -snes_ls <basic,basicnormnorms,quadratic>

  Notes: Solves F(x) - b = 0 using x^{n+1} = x^{n} - lambda (F(x^n) - b) where lambda is obtained either SNESLineSearchSetDamping(), -snes_damping or 
     a line search. 

     This uses no derivative information thus will be much slower then Newton's method obtained with -snes_type ls

.seealso:  SNESCreate(), SNES, SNESSetType(), SNESLS, SNESTR
M*/
EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "SNESCreate_Richardson"
PetscErrorCode  SNESCreate_Richardson(SNES snes)
{
  SNES_Richardson   *neP;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  snes->ops->destroy	     = SNESDestroy_Richardson;
  snes->ops->setup	     = SNESSetUp_Richardson;
  snes->ops->setfromoptions  = SNESSetFromOptions_Richardson;
  snes->ops->view            = SNESView_Richardson;
  snes->ops->solve	     = SNESSolve_Richardson;
  snes->ops->reset           = SNESReset_Richardson;

  snes->usesksp              = PETSC_FALSE;
  snes->usespc               = PETSC_TRUE;

  ierr = PetscNewLog(snes, SNES_Richardson, &neP);CHKERRQ(ierr);
  snes->data = (void*) neP;
  neP->maxstep	     = 1.e8;
  neP->steptol       = 1.e-12;
  neP->type          = SNES_LS_QUADRATIC;
  neP->damping	     = 1.0;
  neP->LineSearch    = SNESLineSearchQuadratic_Richardson;
  neP->lsP           = PETSC_NULL;
  neP->postcheckstep = PETSC_NULL;
  neP->postcheck     = PETSC_NULL;
  neP->precheckstep  = PETSC_NULL;
  neP->precheck      = PETSC_NULL;

  ierr = PetscObjectComposeFunctionDynamic((PetscObject)snes,"SNESLineSearchSetMonitor_C","SNESLineSearchSetMonitor_Richardson",SNESLineSearchSetMonitor_Richardson);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)snes,"SNESLineSearchSet_C","SNESLineSearchSet_Richardson",SNESLineSearchSet_Richardson);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)snes,"SNESLineSearchSetPostCheck_C","SNESLineSearchSetPostCheck_Richardson",SNESLineSearchSetPostCheck_Richardson);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)snes,"SNESLineSearchSetPreCheck_C","SNESLineSearchSetPreCheck_Richardson",SNESLineSearchSetPreCheck_Richardson);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END

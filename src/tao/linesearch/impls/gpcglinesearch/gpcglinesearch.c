#include <petsc/private/taolinesearchimpl.h>
#include <../src/tao/linesearch/impls/gpcglinesearch/gpcglinesearch.h>

/* ---------------------------------------------------------- */

static PetscErrorCode TaoLineSearchDestroy_GPCG(TaoLineSearch ls)
{
  TaoLineSearch_GPCG *ctx = (TaoLineSearch_GPCG *)ls->data;

  PetscFunctionBegin;
  CHKERRQ(VecDestroy(&ctx->W1));
  CHKERRQ(VecDestroy(&ctx->W2));
  CHKERRQ(VecDestroy(&ctx->Gold));
  CHKERRQ(VecDestroy(&ctx->x));
  CHKERRQ(PetscFree(ls->data));
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/
static PetscErrorCode TaoLineSearchView_GPCG(TaoLineSearch ls, PetscViewer viewer)
{
  PetscBool      isascii;

  PetscFunctionBegin;
  CHKERRQ(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&isascii));
  if (isascii) {
    CHKERRQ(PetscViewerASCIIPrintf(viewer," GPCG Line search"));
  }
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/
static PetscErrorCode TaoLineSearchApply_GPCG(TaoLineSearch ls, Vec x, PetscReal *f, Vec g, Vec s)
{
  TaoLineSearch_GPCG *neP = (TaoLineSearch_GPCG *)ls->data;
  PetscInt           i;
  PetscBool          g_computed=PETSC_FALSE; /* to prevent extra gradient computation */
  PetscReal          d1,finit,actred,prered,rho, gdx;

  PetscFunctionBegin;
  /* ls->stepmin - lower bound for step */
  /* ls->stepmax - upper bound for step */
  /* ls->rtol     - relative tolerance for an acceptable step */
  /* ls->ftol     - tolerance for sufficient decrease condition */
  /* ls->gtol     - tolerance for curvature condition */
  /* ls->nfeval   - number of function evaluations */
  /* ls->nfeval   - number of function/gradient evaluations */
  /* ls->max_funcs  - maximum number of function evaluations */

  CHKERRQ(TaoLineSearchMonitor(ls, 0, *f, 0.0));

  ls->reason = TAOLINESEARCH_CONTINUE_ITERATING;
  ls->step = ls->initstep;
  if (!neP->W2) {
    CHKERRQ(VecDuplicate(x,&neP->W2));
    CHKERRQ(VecDuplicate(x,&neP->W1));
    CHKERRQ(VecDuplicate(x,&neP->Gold));
    neP->x = x;
    CHKERRQ(PetscObjectReference((PetscObject)neP->x));
  } else if (x != neP->x) {
    CHKERRQ(VecDestroy(&neP->x));
    CHKERRQ(VecDestroy(&neP->W1));
    CHKERRQ(VecDestroy(&neP->W2));
    CHKERRQ(VecDestroy(&neP->Gold));
    CHKERRQ(VecDuplicate(x,&neP->W1));
    CHKERRQ(VecDuplicate(x,&neP->W2));
    CHKERRQ(VecDuplicate(x,&neP->Gold));
    CHKERRQ(PetscObjectDereference((PetscObject)neP->x));
    neP->x = x;
    CHKERRQ(PetscObjectReference((PetscObject)neP->x));
  }

  CHKERRQ(VecDot(g,s,&gdx));
  if (gdx > 0) {
     CHKERRQ(PetscInfo(ls,"Line search error: search direction is not descent direction. dot(g,s) = %g\n",(double)gdx));
    ls->reason = TAOLINESEARCH_FAILED_ASCENT;
    PetscFunctionReturn(0);
  }
  CHKERRQ(VecCopy(x,neP->W2));
  CHKERRQ(VecCopy(g,neP->Gold));
  if (ls->bounded) {
    /* Compute the smallest steplength that will make one nonbinding variable  equal the bound */
    CHKERRQ(VecStepBoundInfo(x,s,ls->lower,ls->upper,&rho,&actred,&d1));
    ls->step = PetscMin(ls->step,d1);
  }
  rho=0; actred=0;

  if (ls->step < 0) {
    CHKERRQ(PetscInfo(ls,"Line search error: initial step parameter %g< 0\n",(double)ls->step));
    ls->reason = TAOLINESEARCH_HALTED_OTHER;
    PetscFunctionReturn(0);
  }

  /* Initialization */
  finit = *f;
  for (i=0; i< ls->max_funcs; i++) {
    /* Force the step to be within the bounds */
    ls->step = PetscMax(ls->step,ls->stepmin);
    ls->step = PetscMin(ls->step,ls->stepmax);

    CHKERRQ(VecCopy(x,neP->W2));
    CHKERRQ(VecAXPY(neP->W2,ls->step,s));
    if (ls->bounded) {
      /* Make sure new vector is numerically within bounds */
      CHKERRQ(VecMedian(neP->W2,ls->lower,ls->upper,neP->W2));
    }

    /* Gradient is not needed here.  Unless there is a separate
       gradient routine, compute it here anyway to prevent recomputing at
       the end of the line search */
    if (ls->hasobjective) {
      CHKERRQ(TaoLineSearchComputeObjective(ls,neP->W2,f));
      g_computed=PETSC_FALSE;
    } else if (ls->usegts) {
      CHKERRQ(TaoLineSearchComputeObjectiveAndGTS(ls,neP->W2,f,&gdx));
      g_computed=PETSC_FALSE;
    } else {
      CHKERRQ(TaoLineSearchComputeObjectiveAndGradient(ls,neP->W2,f,g));
      g_computed=PETSC_TRUE;
    }

    CHKERRQ(TaoLineSearchMonitor(ls, i+1, *f, ls->step));

    if (0 == i) {
        ls->f_fullstep = *f;
    }

    actred = *f - finit;
    CHKERRQ(VecCopy(neP->W2,neP->W1));
    CHKERRQ(VecAXPY(neP->W1,-1.0,x));    /* W1 = W2 - X */
    CHKERRQ(VecDot(neP->W1,neP->Gold,&prered));

    if (PetscAbsReal(prered)<1.0e-100) prered=1.0e-12;
    rho = actred/prered;

    /*
       If sufficient progress has been obtained, accept the
       point.  Otherwise, backtrack.
    */

    if (actred > 0) {
      CHKERRQ(PetscInfo(ls,"Step resulted in ascent, rejecting.\n"));
      ls->step = (ls->step)/2;
    } else if (rho > ls->ftol) {
      break;
    } else{
      ls->step = (ls->step)/2;
    }

    /* Convergence testing */

    if (ls->step <= ls->stepmin || ls->step >= ls->stepmax) {
      ls->reason = TAOLINESEARCH_HALTED_OTHER;
      CHKERRQ(PetscInfo(ls,"Rounding errors may prevent further progress.  May not be a step satisfying\n"));
      CHKERRQ(PetscInfo(ls,"sufficient decrease and curvature conditions. Tolerances may be too small.\n"));
     break;
    }
    if (ls->step == ls->stepmax) {
      CHKERRQ(PetscInfo(ls,"Step is at the upper bound, stepmax (%g)\n",(double)ls->stepmax));
      ls->reason = TAOLINESEARCH_HALTED_UPPERBOUND;
      break;
    }
    if (ls->step == ls->stepmin) {
      CHKERRQ(PetscInfo(ls,"Step is at the lower bound, stepmin (%g)\n",(double)ls->stepmin));
      ls->reason = TAOLINESEARCH_HALTED_LOWERBOUND;
      break;
    }
    if ((ls->nfeval+ls->nfgeval) >= ls->max_funcs) {
      CHKERRQ(PetscInfo(ls,"Number of line search function evals (%D) > maximum (%D)\n",ls->nfeval+ls->nfgeval,ls->max_funcs));
      ls->reason = TAOLINESEARCH_HALTED_MAXFCN;
      break;
    }
    if ((neP->bracket) && (ls->stepmax - ls->stepmin <= ls->rtol*ls->stepmax)) {
      CHKERRQ(PetscInfo(ls,"Relative width of interval of uncertainty is at most rtol (%g)\n",(double)ls->rtol));
      ls->reason = TAOLINESEARCH_HALTED_RTOL;
      break;
    }
  }
  CHKERRQ(PetscInfo(ls,"%D function evals in line search, step = %g\n",ls->nfeval+ls->nfgeval,(double)ls->step));
  /* set new solution vector and compute gradient if necessary */
  CHKERRQ(VecCopy(neP->W2, x));
  if (ls->reason == TAOLINESEARCH_CONTINUE_ITERATING) {
    ls->reason = TAOLINESEARCH_SUCCESS;
  }
  if (!g_computed) {
    CHKERRQ(TaoLineSearchComputeGradient(ls,x,g));
  }
  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------- */

/*MC
   TAOLINESEARCHGPCG - Special line-search method for the Gradient-Projected Conjugate Gradient (TAOGPCG) algorithm.
   Should not be used with any other algorithm.

   Level: developer

.keywords: Tao, linesearch
M*/
PETSC_EXTERN PetscErrorCode TaoLineSearchCreate_GPCG(TaoLineSearch ls)
{
  TaoLineSearch_GPCG *neP;

  PetscFunctionBegin;
  ls->ftol                = 0.05;
  ls->rtol                = 0.0;
  ls->gtol                = 0.0;
  ls->stepmin             = 1.0e-20;
  ls->stepmax             = 1.0e+20;
  ls->nfeval              = 0;
  ls->max_funcs           = 30;
  ls->step                = 1.0;

  CHKERRQ(PetscNewLog(ls,&neP));
  neP->bracket            = 0;
  neP->infoc              = 1;
  ls->data = (void*)neP;

  ls->ops->setup = NULL;
  ls->ops->reset = NULL;
  ls->ops->apply = TaoLineSearchApply_GPCG;
  ls->ops->view  = TaoLineSearchView_GPCG;
  ls->ops->destroy = TaoLineSearchDestroy_GPCG;
  ls->ops->setfromoptions = NULL;
  ls->ops->monitor = NULL;
  PetscFunctionReturn(0);
}

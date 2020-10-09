#include <petsc/private/taolinesearchimpl.h>
#include <../src/tao/linesearch/impls/gpcglinesearch/gpcglinesearch.h>

/* ---------------------------------------------------------- */

static PetscErrorCode TaoLineSearchDestroy_GPCG(TaoLineSearch ls)
{
  PetscErrorCode     ierr;
  TaoLineSearch_GPCG *ctx = (TaoLineSearch_GPCG *)ls->data;

  PetscFunctionBegin;
  ierr = VecDestroy(&ctx->W1);CHKERRQ(ierr);
  ierr = VecDestroy(&ctx->W2);CHKERRQ(ierr);
  ierr = VecDestroy(&ctx->Gold);CHKERRQ(ierr);
  ierr = VecDestroy(&ctx->x);CHKERRQ(ierr);
  ierr = PetscFree(ls->data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


/*------------------------------------------------------------*/
static PetscErrorCode TaoLineSearchView_GPCG(TaoLineSearch ls, PetscViewer viewer)
{
  PetscBool      isascii;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&isascii);CHKERRQ(ierr);
  if (isascii) {
    ierr = PetscViewerASCIIPrintf(viewer," GPCG Line search");CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/
static PetscErrorCode TaoLineSearchApply_GPCG(TaoLineSearch ls, Vec x, PetscReal *f, Vec g, Vec s)
{
  TaoLineSearch_GPCG *neP = (TaoLineSearch_GPCG *)ls->data;
  PetscErrorCode     ierr;
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

  ierr = TaoLineSearchMonitor(ls, 0, *f, 0.0);CHKERRQ(ierr);

  ls->reason = TAOLINESEARCH_CONTINUE_ITERATING;
  ls->step = ls->initstep;
  if (!neP->W2) {
    ierr = VecDuplicate(x,&neP->W2);CHKERRQ(ierr);
    ierr = VecDuplicate(x,&neP->W1);CHKERRQ(ierr);
    ierr = VecDuplicate(x,&neP->Gold);CHKERRQ(ierr);
    neP->x = x;
    ierr = PetscObjectReference((PetscObject)neP->x);CHKERRQ(ierr);
  } else if (x != neP->x) {
    ierr = VecDestroy(&neP->x);CHKERRQ(ierr);
    ierr = VecDestroy(&neP->W1);CHKERRQ(ierr);
    ierr = VecDestroy(&neP->W2);CHKERRQ(ierr);
    ierr = VecDestroy(&neP->Gold);CHKERRQ(ierr);
    ierr = VecDuplicate(x,&neP->W1);CHKERRQ(ierr);
    ierr = VecDuplicate(x,&neP->W2);CHKERRQ(ierr);
    ierr = VecDuplicate(x,&neP->Gold);CHKERRQ(ierr);
    ierr = PetscObjectDereference((PetscObject)neP->x);CHKERRQ(ierr);
    neP->x = x;
    ierr = PetscObjectReference((PetscObject)neP->x);CHKERRQ(ierr);
  }

  ierr = VecDot(g,s,&gdx);CHKERRQ(ierr);
  if (gdx > 0) {
     ierr = PetscInfo1(ls,"Line search error: search direction is not descent direction. dot(g,s) = %g\n",(double)gdx);CHKERRQ(ierr);
    ls->reason = TAOLINESEARCH_FAILED_ASCENT;
    PetscFunctionReturn(0);
  }
  ierr = VecCopy(x,neP->W2);CHKERRQ(ierr);
  ierr = VecCopy(g,neP->Gold);CHKERRQ(ierr);
  if (ls->bounded) {
    /* Compute the smallest steplength that will make one nonbinding variable  equal the bound */
    ierr = VecStepBoundInfo(x,s,ls->lower,ls->upper,&rho,&actred,&d1);CHKERRQ(ierr);
    ls->step = PetscMin(ls->step,d1);
  }
  rho=0; actred=0;

  if (ls->step < 0) {
    ierr = PetscInfo1(ls,"Line search error: initial step parameter %g< 0\n",(double)ls->step);CHKERRQ(ierr);
    ls->reason = TAOLINESEARCH_HALTED_OTHER;
    PetscFunctionReturn(0);
  }

  /* Initialization */
  finit = *f;
  for (i=0; i< ls->max_funcs; i++) {
    /* Force the step to be within the bounds */
    ls->step = PetscMax(ls->step,ls->stepmin);
    ls->step = PetscMin(ls->step,ls->stepmax);

    ierr = VecCopy(x,neP->W2);CHKERRQ(ierr);
    ierr = VecAXPY(neP->W2,ls->step,s);CHKERRQ(ierr);
    if (ls->bounded) {
      /* Make sure new vector is numerically within bounds */
      ierr = VecMedian(neP->W2,ls->lower,ls->upper,neP->W2);CHKERRQ(ierr);
    }

    /* Gradient is not needed here.  Unless there is a separate
       gradient routine, compute it here anyway to prevent recomputing at
       the end of the line search */
    if (ls->hasobjective) {
      ierr = TaoLineSearchComputeObjective(ls,neP->W2,f);CHKERRQ(ierr);
      g_computed=PETSC_FALSE;
    } else if (ls->usegts){
      ierr = TaoLineSearchComputeObjectiveAndGTS(ls,neP->W2,f,&gdx);CHKERRQ(ierr);
      g_computed=PETSC_FALSE;
    } else {
      ierr = TaoLineSearchComputeObjectiveAndGradient(ls,neP->W2,f,g);CHKERRQ(ierr);
      g_computed=PETSC_TRUE;
    }

    ierr = TaoLineSearchMonitor(ls, i+1, *f, ls->step);CHKERRQ(ierr);

    if (0 == i) {
        ls->f_fullstep = *f;
    }

    actred = *f - finit;
    ierr = VecCopy(neP->W2,neP->W1);CHKERRQ(ierr);
    ierr = VecAXPY(neP->W1,-1.0,x);CHKERRQ(ierr);    /* W1 = W2 - X */
    ierr = VecDot(neP->W1,neP->Gold,&prered);CHKERRQ(ierr);

    if (PetscAbsReal(prered)<1.0e-100) prered=1.0e-12;
    rho = actred/prered;

    /*
       If sufficient progress has been obtained, accept the
       point.  Otherwise, backtrack.
    */

    if (actred > 0) {
      ierr = PetscInfo(ls,"Step resulted in ascent, rejecting.\n");CHKERRQ(ierr);
      ls->step = (ls->step)/2;
    } else if (rho > ls->ftol){
      break;
    } else{
      ls->step = (ls->step)/2;
    }

    /* Convergence testing */

    if (ls->step <= ls->stepmin || ls->step >= ls->stepmax) {
      ls->reason = TAOLINESEARCH_HALTED_OTHER;
      ierr = PetscInfo(ls,"Rounding errors may prevent further progress.  May not be a step satisfying\n");CHKERRQ(ierr);
      ierr = PetscInfo(ls,"sufficient decrease and curvature conditions. Tolerances may be too small.\n");CHKERRQ(ierr);
     break;
    }
    if (ls->step == ls->stepmax) {
      ierr = PetscInfo1(ls,"Step is at the upper bound, stepmax (%g)\n",(double)ls->stepmax);CHKERRQ(ierr);
      ls->reason = TAOLINESEARCH_HALTED_UPPERBOUND;
      break;
    }
    if (ls->step == ls->stepmin) {
      ierr = PetscInfo1(ls,"Step is at the lower bound, stepmin (%g)\n",(double)ls->stepmin);CHKERRQ(ierr);
      ls->reason = TAOLINESEARCH_HALTED_LOWERBOUND;
      break;
    }
    if ((ls->nfeval+ls->nfgeval) >= ls->max_funcs) {
      ierr = PetscInfo2(ls,"Number of line search function evals (%D) > maximum (%D)\n",ls->nfeval+ls->nfgeval,ls->max_funcs);CHKERRQ(ierr);
      ls->reason = TAOLINESEARCH_HALTED_MAXFCN;
      break;
    }
    if ((neP->bracket) && (ls->stepmax - ls->stepmin <= ls->rtol*ls->stepmax)){
      ierr = PetscInfo1(ls,"Relative width of interval of uncertainty is at most rtol (%g)\n",(double)ls->rtol);CHKERRQ(ierr);
      ls->reason = TAOLINESEARCH_HALTED_RTOL;
      break;
    }
  }
  ierr = PetscInfo2(ls,"%D function evals in line search, step = %g\n",ls->nfeval+ls->nfgeval,(double)ls->step);CHKERRQ(ierr);
  /* set new solution vector and compute gradient if necessary */
  ierr = VecCopy(neP->W2, x);CHKERRQ(ierr);
  if (ls->reason == TAOLINESEARCH_CONTINUE_ITERATING) {
    ls->reason = TAOLINESEARCH_SUCCESS;
  }
  if (!g_computed) {
    ierr = TaoLineSearchComputeGradient(ls,x,g);CHKERRQ(ierr);
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
  PetscErrorCode     ierr;
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

  ierr = PetscNewLog(ls,&neP);CHKERRQ(ierr);
  neP->bracket            = 0;
  neP->infoc              = 1;
  ls->data = (void*)neP;

  ls->ops->setup = NULL;
  ls->ops->reset = NULL;
  ls->ops->apply = TaoLineSearchApply_GPCG;
  ls->ops->view  = TaoLineSearchView_GPCG;
  ls->ops->destroy = TaoLineSearchDestroy_GPCG;
  ls->ops->setfromoptions = NULL;
  PetscFunctionReturn(0);
}

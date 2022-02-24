
#include <petsc/private/taolinesearchimpl.h>
#include <../src/tao/linesearch/impls/owarmijo/owarmijo.h>

#define REPLACE_FIFO 1
#define REPLACE_MRU  2

#define REFERENCE_MAX  1
#define REFERENCE_AVE  2
#define REFERENCE_MEAN 3

static PetscErrorCode ProjWork_OWLQN(Vec w,Vec x,Vec gv,PetscReal *gdx)
{
  const PetscReal *xptr,*gptr;
  PetscReal       *wptr;
  PetscInt        low,high,low1,high1,low2,high2,i;

  PetscFunctionBegin;
  CHKERRQ(VecGetOwnershipRange(w,&low,&high));
  CHKERRQ(VecGetOwnershipRange(x,&low1,&high1));
  CHKERRQ(VecGetOwnershipRange(gv,&low2,&high2));

  *gdx=0.0;
  CHKERRQ(VecGetArray(w,&wptr));
  CHKERRQ(VecGetArrayRead(x,&xptr));
  CHKERRQ(VecGetArrayRead(gv,&gptr));

  for (i=0;i<high-low;i++) {
    if (xptr[i]*wptr[i]<0.0) wptr[i]=0.0;
    *gdx = *gdx + gptr[i]*(wptr[i]-xptr[i]);
  }
  CHKERRQ(VecRestoreArray(w,&wptr));
  CHKERRQ(VecRestoreArrayRead(x,&xptr));
  CHKERRQ(VecRestoreArrayRead(gv,&gptr));
  PetscFunctionReturn(0);
}

static PetscErrorCode TaoLineSearchDestroy_OWArmijo(TaoLineSearch ls)
{
  TaoLineSearch_OWARMIJO *armP = (TaoLineSearch_OWARMIJO *)ls->data;

  PetscFunctionBegin;
  CHKERRQ(PetscFree(armP->memory));
  if (armP->x) {
    CHKERRQ(PetscObjectDereference((PetscObject)armP->x));
  }
  CHKERRQ(VecDestroy(&armP->work));
  CHKERRQ(PetscFree(ls->data));
  PetscFunctionReturn(0);
}

static PetscErrorCode TaoLineSearchSetFromOptions_OWArmijo(PetscOptionItems *PetscOptionsObject,TaoLineSearch ls)
{
  TaoLineSearch_OWARMIJO *armP = (TaoLineSearch_OWARMIJO *)ls->data;

  PetscFunctionBegin;
  CHKERRQ(PetscOptionsHead(PetscOptionsObject,"OWArmijo linesearch options"));
  CHKERRQ(PetscOptionsReal("-tao_ls_OWArmijo_alpha", "initial reference constant", "", armP->alpha, &armP->alpha,NULL));
  CHKERRQ(PetscOptionsReal("-tao_ls_OWArmijo_beta_inf", "decrease constant one", "", armP->beta_inf, &armP->beta_inf,NULL));
  CHKERRQ(PetscOptionsReal("-tao_ls_OWArmijo_beta", "decrease constant", "", armP->beta, &armP->beta,NULL));
  CHKERRQ(PetscOptionsReal("-tao_ls_OWArmijo_sigma", "acceptance constant", "", armP->sigma, &armP->sigma,NULL));
  CHKERRQ(PetscOptionsInt("-tao_ls_OWArmijo_memory_size", "number of historical elements", "", armP->memorySize, &armP->memorySize,NULL));
  CHKERRQ(PetscOptionsInt("-tao_ls_OWArmijo_reference_policy", "policy for updating reference value", "", armP->referencePolicy, &armP->referencePolicy,NULL));
  CHKERRQ(PetscOptionsInt("-tao_ls_OWArmijo_replacement_policy", "policy for updating memory", "", armP->replacementPolicy, &armP->replacementPolicy,NULL));
  CHKERRQ(PetscOptionsBool("-tao_ls_OWArmijo_nondescending","Use nondescending OWArmijo algorithm","",armP->nondescending,&armP->nondescending,NULL));
  CHKERRQ(PetscOptionsTail());
  PetscFunctionReturn(0);
}

static PetscErrorCode TaoLineSearchView_OWArmijo(TaoLineSearch ls, PetscViewer pv)
{
  TaoLineSearch_OWARMIJO *armP = (TaoLineSearch_OWARMIJO *)ls->data;
  PetscBool              isascii;

  PetscFunctionBegin;
  CHKERRQ(PetscObjectTypeCompare((PetscObject)pv, PETSCVIEWERASCII, &isascii));
  if (isascii) {
    CHKERRQ(PetscViewerASCIIPrintf(pv,"  OWArmijo linesearch",armP->alpha));
    if (armP->nondescending) {
      CHKERRQ(PetscViewerASCIIPrintf(pv, " (nondescending)"));
    }
    CHKERRQ(PetscViewerASCIIPrintf(pv,": alpha=%g beta=%g ",(double)armP->alpha,(double)armP->beta));
    CHKERRQ(PetscViewerASCIIPrintf(pv,"sigma=%g ",(double)armP->sigma));
    CHKERRQ(PetscViewerASCIIPrintf(pv,"memsize=%D\n",armP->memorySize));
  }
  PetscFunctionReturn(0);
}

/* @ TaoApply_OWArmijo - This routine performs a linesearch. It
   backtracks until the (nonmonotone) OWArmijo conditions are satisfied.

   Input Parameters:
+  tao - TAO_SOLVER context
.  X - current iterate (on output X contains new iterate, X + step*S)
.  S - search direction
.  f - merit function evaluated at X
.  G - gradient of merit function evaluated at X
.  W - work vector
-  step - initial estimate of step length

   Output parameters:
+  f - merit function evaluated at new iterate, X + step*S
.  G - gradient of merit function evaluated at new iterate, X + step*S
.  X - new iterate
-  step - final step length

   Info is set to one of:
.   0 - the line search succeeds; the sufficient decrease
   condition and the directional derivative condition hold

   negative number if an input parameter is invalid
-   -1 -  step < 0

   positive number > 1 if the line search otherwise terminates
+    1 -  Step is at the lower bound, stepmin.
@ */
static PetscErrorCode TaoLineSearchApply_OWArmijo(TaoLineSearch ls, Vec x, PetscReal *f, Vec g, Vec s)
{
  TaoLineSearch_OWARMIJO *armP = (TaoLineSearch_OWARMIJO *)ls->data;
  PetscInt               i, its=0;
  PetscReal              fact, ref, gdx;
  PetscInt               idx;
  PetscBool              g_computed=PETSC_FALSE; /* to prevent extra gradient computation */
  Vec                    g_old;
  PetscReal              owlqn_minstep=0.005;
  PetscReal              partgdx;
  MPI_Comm               comm;

  PetscFunctionBegin;
  CHKERRQ(PetscObjectGetComm((PetscObject)ls,&comm));
  fact = 0.0;
  ls->nfeval=0;
  ls->reason = TAOLINESEARCH_CONTINUE_ITERATING;
  if (!armP->work) {
    CHKERRQ(VecDuplicate(x,&armP->work));
    armP->x = x;
    CHKERRQ(PetscObjectReference((PetscObject)armP->x));
  } else if (x != armP->x) {
    CHKERRQ(VecDestroy(&armP->work));
    CHKERRQ(VecDuplicate(x,&armP->work));
    CHKERRQ(PetscObjectDereference((PetscObject)armP->x));
    armP->x = x;
    CHKERRQ(PetscObjectReference((PetscObject)armP->x));
  }

  CHKERRQ(TaoLineSearchMonitor(ls, 0, *f, 0.0));

  /* Check linesearch parameters */
  if (armP->alpha < 1) {
    CHKERRQ(PetscInfo(ls,"OWArmijo line search error: alpha (%g) < 1\n", (double)armP->alpha));
    ls->reason=TAOLINESEARCH_FAILED_BADPARAMETER;
  } else if ((armP->beta <= 0) || (armP->beta >= 1)) {
    CHKERRQ(PetscInfo(ls,"OWArmijo line search error: beta (%g) invalid\n", (double)armP->beta));
    ls->reason=TAOLINESEARCH_FAILED_BADPARAMETER;
  } else if ((armP->beta_inf <= 0) || (armP->beta_inf >= 1)) {
    CHKERRQ(PetscInfo(ls,"OWArmijo line search error: beta_inf (%g) invalid\n", (double)armP->beta_inf));
    ls->reason=TAOLINESEARCH_FAILED_BADPARAMETER;
  } else if ((armP->sigma <= 0) || (armP->sigma >= 0.5)) {
    CHKERRQ(PetscInfo(ls,"OWArmijo line search error: sigma (%g) invalid\n", (double)armP->sigma));
    ls->reason=TAOLINESEARCH_FAILED_BADPARAMETER;
  } else if (armP->memorySize < 1) {
    CHKERRQ(PetscInfo(ls,"OWArmijo line search error: memory_size (%D) < 1\n", armP->memorySize));
    ls->reason=TAOLINESEARCH_FAILED_BADPARAMETER;
  }  else if ((armP->referencePolicy != REFERENCE_MAX) && (armP->referencePolicy != REFERENCE_AVE) && (armP->referencePolicy != REFERENCE_MEAN)) {
    CHKERRQ(PetscInfo(ls,"OWArmijo line search error: reference_policy invalid\n"));
    ls->reason=TAOLINESEARCH_FAILED_BADPARAMETER;
  } else if ((armP->replacementPolicy != REPLACE_FIFO) && (armP->replacementPolicy != REPLACE_MRU)) {
    CHKERRQ(PetscInfo(ls,"OWArmijo line search error: replacement_policy invalid\n"));
    ls->reason=TAOLINESEARCH_FAILED_BADPARAMETER;
  } else if (PetscIsInfOrNanReal(*f)) {
    CHKERRQ(PetscInfo(ls,"OWArmijo line search error: initial function inf or nan\n"));
    ls->reason=TAOLINESEARCH_FAILED_BADPARAMETER;
  }

  if (ls->reason != TAOLINESEARCH_CONTINUE_ITERATING) PetscFunctionReturn(0);

  /* Check to see of the memory has been allocated.  If not, allocate
     the historical array and populate it with the initial function
     values. */
  if (!armP->memory) {
    CHKERRQ(PetscMalloc1(armP->memorySize, &armP->memory));
  }

  if (!armP->memorySetup) {
    for (i = 0; i < armP->memorySize; i++) {
      armP->memory[i] = armP->alpha*(*f);
    }
    armP->current = 0;
    armP->lastReference = armP->memory[0];
    armP->memorySetup=PETSC_TRUE;
  }

  /* Calculate reference value (MAX) */
  ref = armP->memory[0];
  idx = 0;

  for (i = 1; i < armP->memorySize; i++) {
    if (armP->memory[i] > ref) {
      ref = armP->memory[i];
      idx = i;
    }
  }

  if (armP->referencePolicy == REFERENCE_AVE) {
    ref = 0;
    for (i = 0; i < armP->memorySize; i++) {
      ref += armP->memory[i];
    }
    ref = ref / armP->memorySize;
    ref = PetscMax(ref, armP->memory[armP->current]);
  } else if (armP->referencePolicy == REFERENCE_MEAN) {
    ref = PetscMin(ref, 0.5*(armP->lastReference + armP->memory[armP->current]));
  }

  if (armP->nondescending) {
    fact = armP->sigma;
  }

  CHKERRQ(VecDuplicate(g,&g_old));
  CHKERRQ(VecCopy(g,g_old));

  ls->step = ls->initstep;
  while (ls->step >= owlqn_minstep && ls->nfeval < ls->max_funcs) {
    /* Calculate iterate */
    ++its;
    CHKERRQ(VecCopy(x,armP->work));
    CHKERRQ(VecAXPY(armP->work,ls->step,s));

    partgdx=0.0;
    CHKERRQ(ProjWork_OWLQN(armP->work,x,g_old,&partgdx));
    CHKERRMPI(MPIU_Allreduce(&partgdx,&gdx,1,MPIU_REAL,MPIU_SUM,comm));

    /* Check the condition of gdx */
    if (PetscIsInfOrNanReal(gdx)) {
      CHKERRQ(PetscInfo(ls,"Initial Line Search step * g is Inf or Nan (%g)\n",(double)gdx));
      ls->reason=TAOLINESEARCH_FAILED_INFORNAN;
      PetscFunctionReturn(0);
    }
    if (gdx >= 0.0) {
      CHKERRQ(PetscInfo(ls,"Initial Line Search step is not descent direction (g's=%g)\n",(double)gdx));
      ls->reason = TAOLINESEARCH_FAILED_ASCENT;
      PetscFunctionReturn(0);
    }

    /* Calculate function at new iterate */
    CHKERRQ(TaoLineSearchComputeObjectiveAndGradient(ls,armP->work,f,g));
    g_computed=PETSC_TRUE;

    CHKERRQ(TaoLineSearchMonitor(ls, its, *f, ls->step));

    if (ls->step == ls->initstep) {
      ls->f_fullstep = *f;
    }

    if (PetscIsInfOrNanReal(*f)) {
      ls->step *= armP->beta_inf;
    } else {
      /* Check descent condition */
      if (armP->nondescending && *f <= ref - ls->step*fact*ref) break;
      if (!armP->nondescending && *f <= ref + armP->sigma * gdx) break;
      ls->step *= armP->beta;
    }
  }
  CHKERRQ(VecDestroy(&g_old));

  /* Check termination */
  if (PetscIsInfOrNanReal(*f)) {
    CHKERRQ(PetscInfo(ls, "Function is inf or nan.\n"));
    ls->reason = TAOLINESEARCH_FAILED_BADPARAMETER;
  } else if (ls->step < owlqn_minstep) {
    CHKERRQ(PetscInfo(ls, "Step length is below tolerance.\n"));
    ls->reason = TAOLINESEARCH_HALTED_RTOL;
  } else if (ls->nfeval >= ls->max_funcs) {
    CHKERRQ(PetscInfo(ls, "Number of line search function evals (%D) > maximum allowed (%D)\n",ls->nfeval, ls->max_funcs));
    ls->reason = TAOLINESEARCH_HALTED_MAXFCN;
  }
  if (ls->reason) PetscFunctionReturn(0);

  /* Successful termination, update memory */
  ls->reason = TAOLINESEARCH_SUCCESS;
  armP->lastReference = ref;
  if (armP->replacementPolicy == REPLACE_FIFO) {
    armP->memory[armP->current++] = *f;
    if (armP->current >= armP->memorySize) {
      armP->current = 0;
    }
  } else {
    armP->current = idx;
    armP->memory[idx] = *f;
  }

  /* Update iterate and compute gradient */
  CHKERRQ(VecCopy(armP->work,x));
  if (!g_computed) {
    CHKERRQ(TaoLineSearchComputeGradient(ls, x, g));
  }
  CHKERRQ(PetscInfo(ls, "%D function evals in line search, step = %10.4f\n",ls->nfeval, (double)ls->step));
  PetscFunctionReturn(0);
}

/*MC
   TAOLINESEARCHOWARMIJO - Special line-search type for the Orthant-Wise Limited Quasi-Newton (TAOOWLQN) algorithm.
   Should not be used with any other algorithm.

   Level: developer

.keywords: Tao, linesearch
M*/
PETSC_EXTERN PetscErrorCode TaoLineSearchCreate_OWArmijo(TaoLineSearch ls)
{
  TaoLineSearch_OWARMIJO *armP;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ls,TAOLINESEARCH_CLASSID,1);
  CHKERRQ(PetscNewLog(ls,&armP));

  armP->memory = NULL;
  armP->alpha = 1.0;
  armP->beta = 0.25;
  armP->beta_inf = 0.25;
  armP->sigma = 1e-4;
  armP->memorySize = 1;
  armP->referencePolicy = REFERENCE_MAX;
  armP->replacementPolicy = REPLACE_MRU;
  armP->nondescending=PETSC_FALSE;
  ls->data = (void*)armP;
  ls->initstep = 0.1;
  ls->ops->monitor = NULL;
  ls->ops->setup = NULL;
  ls->ops->reset = NULL;
  ls->ops->apply = TaoLineSearchApply_OWArmijo;
  ls->ops->view = TaoLineSearchView_OWArmijo;
  ls->ops->destroy = TaoLineSearchDestroy_OWArmijo;
  ls->ops->setfromoptions = TaoLineSearchSetFromOptions_OWArmijo;
  PetscFunctionReturn(0);
}

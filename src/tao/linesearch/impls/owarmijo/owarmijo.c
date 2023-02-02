
#include <petsc/private/taolinesearchimpl.h>
#include <../src/tao/linesearch/impls/owarmijo/owarmijo.h>

#define REPLACE_FIFO 1
#define REPLACE_MRU  2

#define REFERENCE_MAX  1
#define REFERENCE_AVE  2
#define REFERENCE_MEAN 3

static PetscErrorCode ProjWork_OWLQN(Vec w, Vec x, Vec gv, PetscReal *gdx)
{
  const PetscReal *xptr, *gptr;
  PetscReal       *wptr;
  PetscInt         low, high, low1, high1, low2, high2, i;

  PetscFunctionBegin;
  PetscCall(VecGetOwnershipRange(w, &low, &high));
  PetscCall(VecGetOwnershipRange(x, &low1, &high1));
  PetscCall(VecGetOwnershipRange(gv, &low2, &high2));

  *gdx = 0.0;
  PetscCall(VecGetArray(w, &wptr));
  PetscCall(VecGetArrayRead(x, &xptr));
  PetscCall(VecGetArrayRead(gv, &gptr));

  for (i = 0; i < high - low; i++) {
    if (xptr[i] * wptr[i] < 0.0) wptr[i] = 0.0;
    *gdx = *gdx + gptr[i] * (wptr[i] - xptr[i]);
  }
  PetscCall(VecRestoreArray(w, &wptr));
  PetscCall(VecRestoreArrayRead(x, &xptr));
  PetscCall(VecRestoreArrayRead(gv, &gptr));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoLineSearchDestroy_OWArmijo(TaoLineSearch ls)
{
  TaoLineSearch_OWARMIJO *armP = (TaoLineSearch_OWARMIJO *)ls->data;

  PetscFunctionBegin;
  PetscCall(PetscFree(armP->memory));
  if (armP->x) PetscCall(PetscObjectDereference((PetscObject)armP->x));
  PetscCall(VecDestroy(&armP->work));
  PetscCall(PetscFree(ls->data));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoLineSearchSetFromOptions_OWArmijo(TaoLineSearch ls, PetscOptionItems *PetscOptionsObject)
{
  TaoLineSearch_OWARMIJO *armP = (TaoLineSearch_OWARMIJO *)ls->data;

  PetscFunctionBegin;
  PetscOptionsHeadBegin(PetscOptionsObject, "OWArmijo linesearch options");
  PetscCall(PetscOptionsReal("-tao_ls_OWArmijo_alpha", "initial reference constant", "", armP->alpha, &armP->alpha, NULL));
  PetscCall(PetscOptionsReal("-tao_ls_OWArmijo_beta_inf", "decrease constant one", "", armP->beta_inf, &armP->beta_inf, NULL));
  PetscCall(PetscOptionsReal("-tao_ls_OWArmijo_beta", "decrease constant", "", armP->beta, &armP->beta, NULL));
  PetscCall(PetscOptionsReal("-tao_ls_OWArmijo_sigma", "acceptance constant", "", armP->sigma, &armP->sigma, NULL));
  PetscCall(PetscOptionsInt("-tao_ls_OWArmijo_memory_size", "number of historical elements", "", armP->memorySize, &armP->memorySize, NULL));
  PetscCall(PetscOptionsInt("-tao_ls_OWArmijo_reference_policy", "policy for updating reference value", "", armP->referencePolicy, &armP->referencePolicy, NULL));
  PetscCall(PetscOptionsInt("-tao_ls_OWArmijo_replacement_policy", "policy for updating memory", "", armP->replacementPolicy, &armP->replacementPolicy, NULL));
  PetscCall(PetscOptionsBool("-tao_ls_OWArmijo_nondescending", "Use nondescending OWArmijo algorithm", "", armP->nondescending, &armP->nondescending, NULL));
  PetscOptionsHeadEnd();
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoLineSearchView_OWArmijo(TaoLineSearch ls, PetscViewer pv)
{
  TaoLineSearch_OWARMIJO *armP = (TaoLineSearch_OWARMIJO *)ls->data;
  PetscBool               isascii;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)pv, PETSCVIEWERASCII, &isascii));
  if (isascii) {
    PetscCall(PetscViewerASCIIPrintf(pv, "  OWArmijo linesearch"));
    if (armP->nondescending) PetscCall(PetscViewerASCIIPrintf(pv, " (nondescending)"));
    PetscCall(PetscViewerASCIIPrintf(pv, ": alpha=%g beta=%g ", (double)armP->alpha, (double)armP->beta));
    PetscCall(PetscViewerASCIIPrintf(pv, "sigma=%g ", (double)armP->sigma));
    PetscCall(PetscViewerASCIIPrintf(pv, "memsize=%" PetscInt_FMT "\n", armP->memorySize));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
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
  PetscInt                i, its = 0;
  PetscReal               fact, ref, gdx;
  PetscInt                idx;
  PetscBool               g_computed = PETSC_FALSE; /* to prevent extra gradient computation */
  Vec                     g_old;
  PetscReal               owlqn_minstep = 0.005;
  PetscReal               partgdx;
  MPI_Comm                comm;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)ls, &comm));
  fact       = 0.0;
  ls->nfeval = 0;
  ls->reason = TAOLINESEARCH_CONTINUE_ITERATING;
  if (!armP->work) {
    PetscCall(VecDuplicate(x, &armP->work));
    armP->x = x;
    PetscCall(PetscObjectReference((PetscObject)armP->x));
  } else if (x != armP->x) {
    PetscCall(VecDestroy(&armP->work));
    PetscCall(VecDuplicate(x, &armP->work));
    PetscCall(PetscObjectDereference((PetscObject)armP->x));
    armP->x = x;
    PetscCall(PetscObjectReference((PetscObject)armP->x));
  }

  PetscCall(TaoLineSearchMonitor(ls, 0, *f, 0.0));

  /* Check linesearch parameters */
  if (armP->alpha < 1) {
    PetscCall(PetscInfo(ls, "OWArmijo line search error: alpha (%g) < 1\n", (double)armP->alpha));
    ls->reason = TAOLINESEARCH_FAILED_BADPARAMETER;
  } else if ((armP->beta <= 0) || (armP->beta >= 1)) {
    PetscCall(PetscInfo(ls, "OWArmijo line search error: beta (%g) invalid\n", (double)armP->beta));
    ls->reason = TAOLINESEARCH_FAILED_BADPARAMETER;
  } else if ((armP->beta_inf <= 0) || (armP->beta_inf >= 1)) {
    PetscCall(PetscInfo(ls, "OWArmijo line search error: beta_inf (%g) invalid\n", (double)armP->beta_inf));
    ls->reason = TAOLINESEARCH_FAILED_BADPARAMETER;
  } else if ((armP->sigma <= 0) || (armP->sigma >= 0.5)) {
    PetscCall(PetscInfo(ls, "OWArmijo line search error: sigma (%g) invalid\n", (double)armP->sigma));
    ls->reason = TAOLINESEARCH_FAILED_BADPARAMETER;
  } else if (armP->memorySize < 1) {
    PetscCall(PetscInfo(ls, "OWArmijo line search error: memory_size (%" PetscInt_FMT ") < 1\n", armP->memorySize));
    ls->reason = TAOLINESEARCH_FAILED_BADPARAMETER;
  } else if ((armP->referencePolicy != REFERENCE_MAX) && (armP->referencePolicy != REFERENCE_AVE) && (armP->referencePolicy != REFERENCE_MEAN)) {
    PetscCall(PetscInfo(ls, "OWArmijo line search error: reference_policy invalid\n"));
    ls->reason = TAOLINESEARCH_FAILED_BADPARAMETER;
  } else if ((armP->replacementPolicy != REPLACE_FIFO) && (armP->replacementPolicy != REPLACE_MRU)) {
    PetscCall(PetscInfo(ls, "OWArmijo line search error: replacement_policy invalid\n"));
    ls->reason = TAOLINESEARCH_FAILED_BADPARAMETER;
  } else if (PetscIsInfOrNanReal(*f)) {
    PetscCall(PetscInfo(ls, "OWArmijo line search error: initial function inf or nan\n"));
    ls->reason = TAOLINESEARCH_FAILED_BADPARAMETER;
  }

  if (ls->reason != TAOLINESEARCH_CONTINUE_ITERATING) PetscFunctionReturn(PETSC_SUCCESS);

  /* Check to see of the memory has been allocated.  If not, allocate
     the historical array and populate it with the initial function
     values. */
  if (!armP->memory) PetscCall(PetscMalloc1(armP->memorySize, &armP->memory));

  if (!armP->memorySetup) {
    for (i = 0; i < armP->memorySize; i++) armP->memory[i] = armP->alpha * (*f);
    armP->current       = 0;
    armP->lastReference = armP->memory[0];
    armP->memorySetup   = PETSC_TRUE;
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
    for (i = 0; i < armP->memorySize; i++) ref += armP->memory[i];
    ref = ref / armP->memorySize;
    ref = PetscMax(ref, armP->memory[armP->current]);
  } else if (armP->referencePolicy == REFERENCE_MEAN) {
    ref = PetscMin(ref, 0.5 * (armP->lastReference + armP->memory[armP->current]));
  }

  if (armP->nondescending) fact = armP->sigma;

  PetscCall(VecDuplicate(g, &g_old));
  PetscCall(VecCopy(g, g_old));

  ls->step = ls->initstep;
  while (ls->step >= owlqn_minstep && ls->nfeval < ls->max_funcs) {
    /* Calculate iterate */
    ++its;
    PetscCall(VecWAXPY(armP->work, ls->step, s, x));

    partgdx = 0.0;
    PetscCall(ProjWork_OWLQN(armP->work, x, g_old, &partgdx));
    PetscCall(MPIU_Allreduce(&partgdx, &gdx, 1, MPIU_REAL, MPIU_SUM, comm));

    /* Check the condition of gdx */
    if (PetscIsInfOrNanReal(gdx)) {
      PetscCall(PetscInfo(ls, "Initial Line Search step * g is Inf or Nan (%g)\n", (double)gdx));
      ls->reason = TAOLINESEARCH_FAILED_INFORNAN;
      PetscFunctionReturn(PETSC_SUCCESS);
    }
    if (gdx >= 0.0) {
      PetscCall(PetscInfo(ls, "Initial Line Search step is not descent direction (g's=%g)\n", (double)gdx));
      ls->reason = TAOLINESEARCH_FAILED_ASCENT;
      PetscFunctionReturn(PETSC_SUCCESS);
    }

    /* Calculate function at new iterate */
    PetscCall(TaoLineSearchComputeObjectiveAndGradient(ls, armP->work, f, g));
    g_computed = PETSC_TRUE;

    PetscCall(TaoLineSearchMonitor(ls, its, *f, ls->step));

    if (ls->step == ls->initstep) ls->f_fullstep = *f;

    if (PetscIsInfOrNanReal(*f)) {
      ls->step *= armP->beta_inf;
    } else {
      /* Check descent condition */
      if (armP->nondescending && *f <= ref - ls->step * fact * ref) break;
      if (!armP->nondescending && *f <= ref + armP->sigma * gdx) break;
      ls->step *= armP->beta;
    }
  }
  PetscCall(VecDestroy(&g_old));

  /* Check termination */
  if (PetscIsInfOrNanReal(*f)) {
    PetscCall(PetscInfo(ls, "Function is inf or nan.\n"));
    ls->reason = TAOLINESEARCH_FAILED_BADPARAMETER;
  } else if (ls->step < owlqn_minstep) {
    PetscCall(PetscInfo(ls, "Step length is below tolerance.\n"));
    ls->reason = TAOLINESEARCH_HALTED_RTOL;
  } else if (ls->nfeval >= ls->max_funcs) {
    PetscCall(PetscInfo(ls, "Number of line search function evals (%" PetscInt_FMT ") > maximum allowed (%" PetscInt_FMT ")\n", ls->nfeval, ls->max_funcs));
    ls->reason = TAOLINESEARCH_HALTED_MAXFCN;
  }
  if (ls->reason) PetscFunctionReturn(PETSC_SUCCESS);

  /* Successful termination, update memory */
  ls->reason          = TAOLINESEARCH_SUCCESS;
  armP->lastReference = ref;
  if (armP->replacementPolicy == REPLACE_FIFO) {
    armP->memory[armP->current++] = *f;
    if (armP->current >= armP->memorySize) armP->current = 0;
  } else {
    armP->current     = idx;
    armP->memory[idx] = *f;
  }

  /* Update iterate and compute gradient */
  PetscCall(VecCopy(armP->work, x));
  if (!g_computed) PetscCall(TaoLineSearchComputeGradient(ls, x, g));
  PetscCall(PetscInfo(ls, "%" PetscInt_FMT " function evals in line search, step = %10.4f\n", ls->nfeval, (double)ls->step));
  PetscFunctionReturn(PETSC_SUCCESS);
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
  PetscValidHeaderSpecific(ls, TAOLINESEARCH_CLASSID, 1);
  PetscCall(PetscNew(&armP));

  armP->memory            = NULL;
  armP->alpha             = 1.0;
  armP->beta              = 0.25;
  armP->beta_inf          = 0.25;
  armP->sigma             = 1e-4;
  armP->memorySize        = 1;
  armP->referencePolicy   = REFERENCE_MAX;
  armP->replacementPolicy = REPLACE_MRU;
  armP->nondescending     = PETSC_FALSE;
  ls->data                = (void *)armP;
  ls->initstep            = 0.1;
  ls->ops->monitor        = NULL;
  ls->ops->setup          = NULL;
  ls->ops->reset          = NULL;
  ls->ops->apply          = TaoLineSearchApply_OWArmijo;
  ls->ops->view           = TaoLineSearchView_OWArmijo;
  ls->ops->destroy        = TaoLineSearchDestroy_OWArmijo;
  ls->ops->setfromoptions = TaoLineSearchSetFromOptions_OWArmijo;
  PetscFunctionReturn(PETSC_SUCCESS);
}

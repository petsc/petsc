#include <petsc/private/taolinesearchimpl.h>
#include <../src/tao/linesearch/impls/armijo/armijo.h>

#define REPLACE_FIFO 1
#define REPLACE_MRU  2

#define REFERENCE_MAX  1
#define REFERENCE_AVE  2
#define REFERENCE_MEAN 3

static PetscErrorCode TaoLineSearchDestroy_Armijo(TaoLineSearch ls)
{
  TaoLineSearch_ARMIJO *armP = (TaoLineSearch_ARMIJO *)ls->data;
  PetscErrorCode       ierr;

  PetscFunctionBegin;
  ierr = PetscFree(armP->memory);CHKERRQ(ierr);
  ierr = VecDestroy(&armP->x);CHKERRQ(ierr);
  ierr = VecDestroy(&armP->work);CHKERRQ(ierr);
  ierr = PetscFree(ls->data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode TaoLineSearchReset_Armijo(TaoLineSearch ls)
{
  TaoLineSearch_ARMIJO *armP = (TaoLineSearch_ARMIJO *)ls->data;
  PetscErrorCode       ierr;

  PetscFunctionBegin;
  ierr = PetscFree(armP->memory);CHKERRQ(ierr);
  armP->memorySetup = PETSC_FALSE;
  PetscFunctionReturn(0);
}

static PetscErrorCode TaoLineSearchSetFromOptions_Armijo(PetscOptionItems *PetscOptionsObject,TaoLineSearch ls)
{
  TaoLineSearch_ARMIJO *armP = (TaoLineSearch_ARMIJO *)ls->data;
  PetscErrorCode       ierr;

  PetscFunctionBegin;
  ierr = PetscOptionsHead(PetscOptionsObject,"Armijo linesearch options");CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_ls_armijo_alpha", "initial reference constant", "", armP->alpha, &armP->alpha,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_ls_armijo_beta_inf", "decrease constant one", "", armP->beta_inf, &armP->beta_inf,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_ls_armijo_beta", "decrease constant", "", armP->beta, &armP->beta,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_ls_armijo_sigma", "acceptance constant", "", armP->sigma, &armP->sigma,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-tao_ls_armijo_memory_size", "number of historical elements", "", armP->memorySize, &armP->memorySize,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-tao_ls_armijo_reference_policy", "policy for updating reference value", "", armP->referencePolicy, &armP->referencePolicy,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-tao_ls_armijo_replacement_policy", "policy for updating memory", "", armP->replacementPolicy, &armP->replacementPolicy,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-tao_ls_armijo_nondescending","Use nondescending armijo algorithm","",armP->nondescending,&armP->nondescending,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode TaoLineSearchView_Armijo(TaoLineSearch ls, PetscViewer pv)
{
  TaoLineSearch_ARMIJO *armP = (TaoLineSearch_ARMIJO *)ls->data;
  PetscBool            isascii;
  PetscErrorCode       ierr;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)pv, PETSCVIEWERASCII, &isascii);CHKERRQ(ierr);
  if (isascii) {
    ierr = PetscViewerASCIIPrintf(pv,"  Armijo linesearch",armP->alpha);CHKERRQ(ierr);
    if (armP->nondescending) {
      ierr = PetscViewerASCIIPrintf(pv, " (nondescending)");CHKERRQ(ierr);
    }
    if (ls->bounded) {
      ierr = PetscViewerASCIIPrintf(pv," (projected)");CHKERRQ(ierr);
    }
    ierr = PetscViewerASCIIPrintf(pv,": alpha=%g beta=%g ",(double)armP->alpha,(double)armP->beta);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(pv,"sigma=%g ",(double)armP->sigma);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(pv,"memsize=%D\n",armP->memorySize);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/* @ TaoApply_Armijo - This routine performs a linesearch. It
   backtracks until the (nonmonotone) Armijo conditions are satisfied.

   Input Parameters:
+  tao - Tao context
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

@ */
static PetscErrorCode TaoLineSearchApply_Armijo(TaoLineSearch ls, Vec x, PetscReal *f, Vec g, Vec s)
{
  TaoLineSearch_ARMIJO *armP = (TaoLineSearch_ARMIJO *)ls->data;
  PetscErrorCode       ierr;
  PetscInt             i,its=0;
  PetscReal            fact, ref, gdx;
  PetscInt             idx;
  PetscBool            g_computed=PETSC_FALSE; /* to prevent extra gradient computation */

  PetscFunctionBegin;
  ierr = TaoLineSearchMonitor(ls, 0, *f, 0.0);CHKERRQ(ierr);

  ls->reason = TAOLINESEARCH_CONTINUE_ITERATING;
  if (!armP->work) {
    ierr = VecDuplicate(x,&armP->work);CHKERRQ(ierr);
    armP->x = x;
    ierr = PetscObjectReference((PetscObject)armP->x);CHKERRQ(ierr);
  } else if (x != armP->x) {
    /* If x has changed, then recreate work */
    ierr = VecDestroy(&armP->work);CHKERRQ(ierr);
    ierr = VecDuplicate(x,&armP->work);CHKERRQ(ierr);
    ierr = PetscObjectDereference((PetscObject)armP->x);CHKERRQ(ierr);
    armP->x = x;
    ierr = PetscObjectReference((PetscObject)armP->x);CHKERRQ(ierr);
  }

  /* Check linesearch parameters */
  if (armP->alpha < 1) {
    ierr = PetscInfo(ls,"Armijo line search error: alpha (%g) < 1\n", (double)armP->alpha);CHKERRQ(ierr);
    ls->reason=TAOLINESEARCH_FAILED_BADPARAMETER;
  } else if ((armP->beta <= 0) || (armP->beta >= 1)) {
    ierr = PetscInfo(ls,"Armijo line search error: beta (%g) invalid\n", (double)armP->beta);CHKERRQ(ierr);
    ls->reason=TAOLINESEARCH_FAILED_BADPARAMETER;
  } else if ((armP->beta_inf <= 0) || (armP->beta_inf >= 1)) {
    ierr = PetscInfo(ls,"Armijo line search error: beta_inf (%g) invalid\n", (double)armP->beta_inf);CHKERRQ(ierr);
    ls->reason=TAOLINESEARCH_FAILED_BADPARAMETER;
  } else if ((armP->sigma <= 0) || (armP->sigma >= 0.5)) {
    ierr = PetscInfo(ls,"Armijo line search error: sigma (%g) invalid\n", (double)armP->sigma);CHKERRQ(ierr);
    ls->reason=TAOLINESEARCH_FAILED_BADPARAMETER;
  } else if (armP->memorySize < 1) {
    ierr = PetscInfo(ls,"Armijo line search error: memory_size (%D) < 1\n", armP->memorySize);CHKERRQ(ierr);
    ls->reason=TAOLINESEARCH_FAILED_BADPARAMETER;
  } else if ((armP->referencePolicy != REFERENCE_MAX) && (armP->referencePolicy != REFERENCE_AVE) && (armP->referencePolicy != REFERENCE_MEAN)) {
    ierr = PetscInfo(ls,"Armijo line search error: reference_policy invalid\n");CHKERRQ(ierr);
    ls->reason=TAOLINESEARCH_FAILED_BADPARAMETER;
  } else if ((armP->replacementPolicy != REPLACE_FIFO) && (armP->replacementPolicy != REPLACE_MRU)) {
    ierr = PetscInfo(ls,"Armijo line search error: replacement_policy invalid\n");CHKERRQ(ierr);
    ls->reason=TAOLINESEARCH_FAILED_BADPARAMETER;
  } else if (PetscIsInfOrNanReal(*f)) {
    ierr = PetscInfo(ls,"Armijo line search error: initial function inf or nan\n");CHKERRQ(ierr);
    ls->reason=TAOLINESEARCH_FAILED_BADPARAMETER;
  }

  if (ls->reason != TAOLINESEARCH_CONTINUE_ITERATING) {
    PetscFunctionReturn(0);
  }

  /* Check to see of the memory has been allocated.  If not, allocate
     the historical array and populate it with the initial function
     values. */
  if (!armP->memory) {
    ierr = PetscMalloc1(armP->memorySize, &armP->memory);CHKERRQ(ierr);
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
  ierr = VecDot(g,s,&gdx);CHKERRQ(ierr);

  if (PetscIsInfOrNanReal(gdx)) {
    ierr = PetscInfo(ls,"Initial Line Search step * g is Inf or Nan (%g)\n",(double)gdx);CHKERRQ(ierr);
    ls->reason=TAOLINESEARCH_FAILED_INFORNAN;
    PetscFunctionReturn(0);
  }
  if (gdx >= 0.0) {
    ierr = PetscInfo(ls,"Initial Line Search step is not descent direction (g's=%g)\n",(double)gdx);CHKERRQ(ierr);
    ls->reason = TAOLINESEARCH_FAILED_ASCENT;
    PetscFunctionReturn(0);
  }

  if (armP->nondescending) {
    fact = armP->sigma;
  } else {
    fact = armP->sigma * gdx;
  }
  ls->step = ls->initstep;
  while (ls->step >= ls->stepmin && (ls->nfeval+ls->nfgeval) < ls->max_funcs) {
    /* Calculate iterate */
    ++its;
    ierr = VecCopy(x,armP->work);CHKERRQ(ierr);
    ierr = VecAXPY(armP->work,ls->step,s);CHKERRQ(ierr);
    if (ls->bounded) {
      ierr = VecMedian(ls->lower,armP->work,ls->upper,armP->work);CHKERRQ(ierr);
    }

    /* Calculate function at new iterate */
    if (ls->hasobjective) {
      ierr = TaoLineSearchComputeObjective(ls,armP->work,f);CHKERRQ(ierr);
      g_computed=PETSC_FALSE;
    } else if (ls->usegts) {
      ierr = TaoLineSearchComputeObjectiveAndGTS(ls,armP->work,f,&gdx);CHKERRQ(ierr);
      g_computed=PETSC_FALSE;
    } else {
      ierr = TaoLineSearchComputeObjectiveAndGradient(ls,armP->work,f,g);CHKERRQ(ierr);
      g_computed=PETSC_TRUE;
    }
    if (ls->step == ls->initstep) {
      ls->f_fullstep = *f;
    }

    ierr = TaoLineSearchMonitor(ls, its, *f, ls->step);CHKERRQ(ierr);

    if (PetscIsInfOrNanReal(*f)) {
      ls->step *= armP->beta_inf;
    } else {
      /* Check descent condition */
      if (armP->nondescending && *f <= ref - ls->step*fact*ref)
        break;
      if (!armP->nondescending && *f <= ref + ls->step*fact) {
        break;
      }

      ls->step *= armP->beta;
    }
  }

  /* Check termination */
  if (PetscIsInfOrNanReal(*f)) {
    ierr = PetscInfo(ls, "Function is inf or nan.\n");CHKERRQ(ierr);
    ls->reason = TAOLINESEARCH_FAILED_INFORNAN;
  } else if (ls->step < ls->stepmin) {
    ierr = PetscInfo(ls, "Step length is below tolerance.\n");CHKERRQ(ierr);
    ls->reason = TAOLINESEARCH_HALTED_RTOL;
  } else if ((ls->nfeval+ls->nfgeval) >= ls->max_funcs) {
    ierr = PetscInfo(ls, "Number of line search function evals (%D) > maximum allowed (%D)\n",ls->nfeval+ls->nfgeval, ls->max_funcs);CHKERRQ(ierr);
    ls->reason = TAOLINESEARCH_HALTED_MAXFCN;
  }
  if (ls->reason) {
    PetscFunctionReturn(0);
  }

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
  ierr = VecCopy(armP->work,x);CHKERRQ(ierr);
  if (!g_computed) {
    ierr = TaoLineSearchComputeGradient(ls, x, g);CHKERRQ(ierr);
  }
  ierr = PetscInfo(ls, "%D function evals in line search, step = %g\n",ls->nfeval, (double)ls->step);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*MC
   TAOLINESEARCHARMIJO - Backtracking line-search that satisfies only the (nonmonotone) Armijo condition
   (i.e., sufficient decrease).

   Armijo line-search type can be selected with "-tao_ls_type armijo".

   Level: developer

.seealso: TaoLineSearchCreate(), TaoLineSearchSetType(), TaoLineSearchApply()

.keywords: Tao, linesearch
M*/
PETSC_EXTERN PetscErrorCode TaoLineSearchCreate_Armijo(TaoLineSearch ls)
{
  TaoLineSearch_ARMIJO *armP;
  PetscErrorCode       ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ls,TAOLINESEARCH_CLASSID,1);
  ierr = PetscNewLog(ls,&armP);CHKERRQ(ierr);

  armP->memory = NULL;
  armP->alpha = 1.0;
  armP->beta = 0.5;
  armP->beta_inf = 0.5;
  armP->sigma = 1e-4;
  armP->memorySize = 1;
  armP->referencePolicy = REFERENCE_MAX;
  armP->replacementPolicy = REPLACE_MRU;
  armP->nondescending=PETSC_FALSE;
  ls->data = (void*)armP;
  ls->initstep = 1.0;
  ls->ops->setup = NULL;
  ls->ops->monitor = NULL;
  ls->ops->apply = TaoLineSearchApply_Armijo;
  ls->ops->view = TaoLineSearchView_Armijo;
  ls->ops->destroy = TaoLineSearchDestroy_Armijo;
  ls->ops->reset = TaoLineSearchReset_Armijo;
  ls->ops->setfromoptions = TaoLineSearchSetFromOptions_Armijo;
  PetscFunctionReturn(0);
}

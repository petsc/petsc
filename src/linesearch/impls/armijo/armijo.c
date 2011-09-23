#include "petscvec.h"
#include "taosolver.h"
#include "private/taolinesearch_impl.h"
#include "armijo.h"

#define REPLACE_FIFO 1
#define REPLACE_MRU  2

#define REFERENCE_MAX  1
#define REFERENCE_AVE  2
#define REFERENCE_MEAN 3

#undef __FUNCT__  
#define __FUNCT__ "TaoLineSearchDestroy_Armijo"
static PetscErrorCode TaoLineSearchDestroy_Armijo(TaoLineSearch ls)
{
  TAOLINESEARCH_ARMIJO_CTX *armP = (TAOLINESEARCH_ARMIJO_CTX *)ls->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;

  if (armP->memory != PETSC_NULL) {
    ierr = PetscFree(armP->memory); CHKERRQ(ierr);
    armP->memory = PETSC_NULL;
  }
  if (armP->x) {
    ierr = PetscObjectDereference((PetscObject)armP->x); CHKERRQ(ierr);
  }
  ierr = VecDestroy(&armP->work); CHKERRQ(ierr);
  ierr = PetscFree(ls->data); CHKERRQ(ierr);
  ls->data = PETSC_NULL;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "TaoLineSearchSetFromOptions_Armijo"
static PetscErrorCode TaoLineSearchSetFromOptions_Armijo(TaoLineSearch ls)
{
  TAOLINESEARCH_ARMIJO_CTX *armP = (TAOLINESEARCH_ARMIJO_CTX *)ls->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscOptionsHead("Armijo linesearch options");CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_ls_armijo_alpha", "initial reference constant", "", armP->alpha, &armP->alpha, 0); CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_ls_armijo_beta_inf", "decrease constant one", "", armP->beta_inf, &armP->beta_inf, 0); CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_ls_armijo_beta", "decrease constant", "", armP->beta, &armP->beta, 0); CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_ls_armijo_sigma", "acceptance constant", "", armP->sigma, &armP->sigma, 0); CHKERRQ(ierr);
  ierr = PetscOptionsInt("-tao_ls_armijo_memory_size", "number of historical elements", "", armP->memorySize, &armP->memorySize, 0); CHKERRQ(ierr);
  ierr = PetscOptionsInt("-tao_ls_armijo_reference_policy", "policy for updating reference value", "", armP->referencePolicy, &armP->referencePolicy, 0); CHKERRQ(ierr);
  ierr = PetscOptionsInt("-tao_ls_armijo_replacement_policy", "policy for updating memory", "", armP->replacementPolicy, &armP->replacementPolicy, 0); CHKERRQ(ierr);
  ierr = PetscOptionsBool("-tao_ls_armijo_nondescending","Use nondescending armijo algorithm","",armP->nondescending,&armP->nondescending, 0); CHKERRQ(ierr);
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "TaoLineSearchView_Armijo"
static PetscErrorCode TaoLineSearchView_Armijo(TaoLineSearch ls, PetscViewer pv)
{
  TAOLINESEARCH_ARMIJO_CTX *armP = (TAOLINESEARCH_ARMIJO_CTX *)ls->data;
  PetscBool isascii;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscTypeCompare((PetscObject)pv, PETSCVIEWERASCII, &isascii); CHKERRQ(ierr);
  if (isascii) {
    ierr = PetscViewerASCIIPrintf(pv,"  maxf=%D, ftol=%G, gtol=%G\n",ls->maxfev, ls->rtol, ls->ftol); CHKERRQ(ierr);
    ierr=PetscViewerASCIIPrintf(pv,"  Armijo linesearch",armP->alpha);CHKERRQ(ierr);
    if (armP->nondescending) {
      ierr = PetscViewerASCIIPrintf(pv, " (nondescending)"); CHKERRQ(ierr);
    }
    ierr=PetscViewerASCIIPrintf(pv,": alpha=%G beta=%G ",armP->alpha,armP->beta);CHKERRQ(ierr);
    ierr=PetscViewerASCIIPrintf(pv,"sigma=%G ",armP->sigma);CHKERRQ(ierr);
    ierr=PetscViewerASCIIPrintf(pv,"memsize=%D\n",armP->memorySize);CHKERRQ(ierr);
  } else {
    SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_SUP,"Viewer type %s not supported for Armijo TaoLineSearch",((PetscObject)pv)->type_name);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "TaoLineSearchApply_Armijo"
/* @ TaoApply_Armijo - This routine performs a linesearch. It
   backtracks until the (nonmonotone) Armijo conditions are satisfied.

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
static PetscErrorCode TaoLineSearchApply_Armijo(TaoLineSearch ls, Vec x, PetscReal *f, Vec g, Vec s)
{
  TAOLINESEARCH_ARMIJO_CTX *armP = (TAOLINESEARCH_ARMIJO_CTX *)ls->data;
  PetscErrorCode ierr;
  PetscInt i;
  PetscReal fact, ref, gdx;
  PetscInt idx;
  PetscBool g_computed=PETSC_FALSE; /* to prevent extra gradient computation */

  PetscFunctionBegin;

  ls->nfeval=0;
  ls->reason = TAOLINESEARCH_CONTINUE_ITERATING;
  if (!armP->work) {
    ierr = VecDuplicate(x,&armP->work); CHKERRQ(ierr);
    armP->x = x;
    ierr = PetscObjectReference((PetscObject)armP->x); CHKERRQ(ierr);
  }
  /* If x has changed, then recreate work */
  else if (x != armP->x) {
    ierr = VecDestroy(&armP->work); CHKERRQ(ierr);
    ierr = VecDuplicate(x,&armP->work); CHKERRQ(ierr);
    ierr = PetscObjectDereference((PetscObject)armP->x); CHKERRQ(ierr);
    armP->x = x;
    ierr = PetscObjectReference((PetscObject)armP->x); CHKERRQ(ierr);
  }

  /* Check linesearch parameters */
  if (armP->alpha < 1) {
    ierr = PetscInfo1(ls,"Armijo line search error: alpha (%G) < 1\n", armP->alpha); CHKERRQ(ierr);
    ls->reason=TAOLINESEARCH_FAILED_BADPARAMETER;
  } 
  
  else if ((armP->beta <= 0) || (armP->beta >= 1)) {
    ierr = PetscInfo1(ls,"Armijo line search error: beta (%G) invalid\n", armP->beta); CHKERRQ(ierr);
    ls->reason=TAOLINESEARCH_FAILED_BADPARAMETER;

  } 
  
  else if ((armP->beta_inf <= 0) || (armP->beta_inf >= 1)) {
    ierr = PetscInfo1(ls,"Armijo line search error: beta_inf (%G) invalid\n", armP->beta_inf); CHKERRQ(ierr);
    ls->reason=TAOLINESEARCH_FAILED_BADPARAMETER;
  } 

  else if ((armP->sigma <= 0) || (armP->sigma >= 0.5)) {
    ierr = PetscInfo1(ls,"Armijo line search error: sigma (%G) invalid\n", armP->sigma); CHKERRQ(ierr);
    ls->reason=TAOLINESEARCH_FAILED_BADPARAMETER;
  } 
  
  else if (armP->memorySize < 1) {
    ierr = PetscInfo1(ls,"Armijo line search error: memory_size (%D) < 1\n", armP->memorySize); CHKERRQ(ierr);
    ls->reason=TAOLINESEARCH_FAILED_BADPARAMETER;
  } 
  
  else if ((armP->referencePolicy != REFERENCE_MAX) &&
      (armP->referencePolicy != REFERENCE_AVE) &&
      (armP->referencePolicy != REFERENCE_MEAN)) {
    ierr = PetscInfo(ls,"Armijo line search error: reference_policy invalid\n"); CHKERRQ(ierr);
    ls->reason=TAOLINESEARCH_FAILED_BADPARAMETER;

  } 
  
  else if ((armP->replacementPolicy != REPLACE_FIFO) && 
      (armP->replacementPolicy != REPLACE_MRU)) {
    ierr = PetscInfo(ls,"Armijo line search error: replacement_policy invalid\n"); CHKERRQ(ierr);
    ls->reason=TAOLINESEARCH_FAILED_BADPARAMETER;
  }
  
  else if (PetscIsInfOrNanReal(*f)) {
    ierr = PetscInfo(ls,"Armijo line search error: initial function inf or nan\n"); CHKERRQ(ierr);
    ls->reason=TAOLINESEARCH_FAILED_BADPARAMETER;
  }

  if (ls->reason != TAOLINESEARCH_CONTINUE_ITERATING) {
    PetscFunctionReturn(0);
  }

  /* Check to see of the memory has been allocated.  If not, allocate
     the historical array and populate it with the initial function
     values. */
  if (armP->memory == PETSC_NULL) {
    ierr = PetscMalloc(sizeof(PetscReal)*armP->memorySize, &armP->memory ); CHKERRQ(ierr);
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
  } 
  else if (armP->referencePolicy == REFERENCE_MEAN) {
    ref = PetscMin(ref, 0.5*(armP->lastReference + armP->memory[armP->current]));
  }
  ierr = VecDot(g,s,&gdx); CHKERRQ(ierr);

  if (PetscIsInfOrNanReal(gdx)) {
    ierr = PetscInfo1(ls,"Initial Line Search step * g is Inf or Nan (%G)\n",gdx); CHKERRQ(ierr);
    ls->reason=TAOLINESEARCH_FAILED_INFORNAN;
    PetscFunctionReturn(0);
  }
  if (gdx >= 0.0) {
    ierr = PetscInfo1(ls,"Initial Line Search step is not descent direction (g's=%G)\n",gdx); CHKERRQ(ierr);
    ls->reason = TAOLINESEARCH_FAILED_ASCENT;
    PetscFunctionReturn(0);
  }
  
  if (armP->nondescending) {
    fact = armP->sigma; 
  } else {
    fact = armP->sigma * gdx;
  }
  ls->step = ls->initstep;
  while (ls->step >= ls->stepmin && ls->nfeval < ls->maxfev) {
    /* Calculate iterate */
    ierr = VecCopy(x,armP->work); CHKERRQ(ierr);
    ierr = VecAXPY(armP->work,ls->step,s); CHKERRQ(ierr);

    /* Calculate function at new iterate */
    if (ls->hasobjective) {
      ierr = TaoLineSearchComputeObjective(ls,armP->work,f); CHKERRQ(ierr);
      g_computed=PETSC_FALSE;
    } else if (ls->usegts) {
      ierr = TaoLineSearchComputeObjectiveAndGTS(ls,armP->work,f,&gdx); CHKERRQ(ierr);
      g_computed=PETSC_FALSE;
    } else {
      ierr = TaoLineSearchComputeObjectiveAndGradient(ls,armP->work,f,g); CHKERRQ(ierr);
      g_computed=PETSC_TRUE;
    }
    if (ls->step == ls->initstep) {
      ls->f_fullstep = *f;
    }

    if (PetscIsInfOrNanReal(*f)) {
      ls->step *= armP->beta_inf;
    }
    else {
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
    ierr = PetscInfo(ls, "Function is inf or nan.\n"); CHKERRQ(ierr);
    ls->reason = TAOLINESEARCH_FAILED_DOMAIN;
  } else if (ls->step < ls->stepmin) {
    ierr = PetscInfo(ls, "Step length is below tolerance.\n"); CHKERRQ(ierr);
    ls->reason = TAOLINESEARCH_FAILED_RTOL;
  } else if (ls->nfeval >= ls->maxfev) {
    ierr = PetscInfo2(ls, "Number of line search function evals (%D) > maximum allowed (%D)\n",ls->nfeval, ls->maxfev); CHKERRQ(ierr);
    ls->reason = TAOLINESEARCH_FAILED_MAXFCN;
  } 
  if (ls->reason) {
    PetscFunctionReturn(0);
  }

  /* Successful termination, update memory */
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
  ierr = VecCopy(armP->work,x); CHKERRQ(ierr);
  if (!g_computed) {
    ierr = TaoLineSearchComputeGradient(ls, x, g); CHKERRQ(ierr);
  }

  /* Finish computations */
  ierr = PetscInfo2(ls, "%D function evals in line search, step = %G\n",ls->nfeval, ls->step); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "TaoLineSearchCreate_Armijo"
PetscErrorCode TaoLineSearchCreate_Armijo(TaoLineSearch ls)
{
  TAOLINESEARCH_ARMIJO_CTX *armP;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ls,TAOLINESEARCH_CLASSID,1);
  ierr = PetscNewLog(ls,TAOLINESEARCH_ARMIJO_CTX, &armP);CHKERRQ(ierr);

  armP->memory = TAO_NULL;
  armP->alpha = 1.0;
  armP->beta = 0.5;
  armP->beta_inf = 0.5;
  armP->sigma = 1e-4;
  armP->memorySize = 1;
  armP->referencePolicy = REFERENCE_MAX;
  armP->replacementPolicy = REPLACE_MRU;
  armP->nondescending=PETSC_FALSE;
  ls->data = (void*)armP;
  ls->initstep=1.0;
  ls->ops->setup=0;
  ls->ops->apply=TaoLineSearchApply_Armijo;
  ls->ops->view = TaoLineSearchView_Armijo;
  ls->ops->destroy = TaoLineSearchDestroy_Armijo;
  ls->ops->setfromoptions = TaoLineSearchSetFromOptions_Armijo;

  PetscFunctionReturn(0);
}
EXTERN_C_END

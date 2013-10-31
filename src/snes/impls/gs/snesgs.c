#include <petsc-private/snesimpl.h>             /*I   "petscsnes.h"   I*/

typedef struct {
  PetscInt  sweeps;     /* number of sweeps through the local subdomain before neighbor communication */
  PetscInt  max_its;    /* maximum iterations of the inner pointblock solver */
  PetscReal rtol;       /* relative tolerance of the inner pointblock solver */
  PetscReal abstol;     /* absolute tolerance of the inner pointblock solver */
  PetscReal stol;       /* step tolerance of the inner pointblock solver */
} SNES_GS;


#undef __FUNCT__
#define __FUNCT__ "SNESGSSetTolerances"
/*@
   SNESGSSetTolerances - Sets various parameters used in convergence tests.

   Logically Collective on SNES

   Input Parameters:
+  snes - the SNES context
.  abstol - absolute convergence tolerance
.  rtol - relative convergence tolerance
.  stol -  convergence tolerance in terms of the norm of the change in the solution between steps,  || delta x || < stol*|| x ||
-  maxit - maximum number of iterations


   Options Database Keys:
+    -snes_gs_atol <abstol> - Sets abstol
.    -snes_gs_rtol <rtol> - Sets rtol
.    -snes_gs_stol <stol> - Sets stol
-    -snes_max_it <maxit> - Sets maxit

   Level: intermediate

.keywords: SNES, nonlinear, gauss-seidel, set, convergence, tolerances

.seealso: SNESSetTrustRegionTolerance()
@*/
PetscErrorCode  SNESGSSetTolerances(SNES snes,PetscReal abstol,PetscReal rtol,PetscReal stol,PetscInt maxit)
{
  SNES_GS *gs = (SNES_GS*)snes->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);

  if (abstol != PETSC_DEFAULT) {
    if (abstol < 0.0) SETERRQ1(PetscObjectComm((PetscObject)snes),PETSC_ERR_ARG_OUTOFRANGE,"Absolute tolerance %G must be non-negative",abstol);
    gs->abstol = abstol;
  }
  if (rtol != PETSC_DEFAULT) {
    if (rtol < 0.0 || 1.0 <= rtol) SETERRQ1(PetscObjectComm((PetscObject)snes),PETSC_ERR_ARG_OUTOFRANGE,"Relative tolerance %G must be non-negative and less than 1.0",rtol);
    gs->rtol = rtol;
  }
  if (stol != PETSC_DEFAULT) {
    if (stol < 0.0) SETERRQ1(PetscObjectComm((PetscObject)snes),PETSC_ERR_ARG_OUTOFRANGE,"Step tolerance %G must be non-negative",stol);
    gs->stol = stol;
  }
  if (maxit != PETSC_DEFAULT) {
    if (maxit < 0) SETERRQ1(PetscObjectComm((PetscObject)snes),PETSC_ERR_ARG_OUTOFRANGE,"Maximum number of iterations %D must be non-negative",maxit);
    gs->max_its = maxit;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESGSGetTolerances"
/*@
   SNESGSGetTolerances - Gets various parameters used in convergence tests.

   Not Collective

   Input Parameters:
+  snes - the SNES context
.  atol - absolute convergence tolerance
.  rtol - relative convergence tolerance
.  stol -  convergence tolerance in terms of the norm
           of the change in the solution between steps
-  maxit - maximum number of iterations

   Notes:
   The user can specify NULL for any parameter that is not needed.

   Level: intermediate

.keywords: SNES, nonlinear, get, convergence, tolerances

.seealso: SNESSetTolerances()
@*/
PetscErrorCode  SNESGSGetTolerances(SNES snes,PetscReal *atol,PetscReal *rtol,PetscReal *stol,PetscInt *maxit)
{
  SNES_GS *gs = (SNES_GS*)snes->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  if (atol) *atol = gs->abstol;
  if (rtol) *rtol = gs->rtol;
  if (stol) *stol = gs->stol;
  if (maxit) *maxit = gs->max_its;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESGSSetSweeps"
/*@
   SNESGSSetSweeps - Sets the number of sweeps of GS to use.

   Input Parameters:
+  snes   - the SNES context
-  sweeps  - the number of sweeps of GS to perform.

   Level: intermediate

.keywords: SNES, nonlinear, set, Gauss-Siedel

.seealso: SNESSetGS(), SNESGetGS(), SNESSetPC(), SNESGSGetSweeps()
@*/

PetscErrorCode SNESGSSetSweeps(SNES snes, PetscInt sweeps)
{
  SNES_GS *gs = (SNES_GS*)snes->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  gs->sweeps = sweeps;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESGSGetSweeps"
/*@
   SNESGSGetSweeps - Gets the number of sweeps GS will use.

   Input Parameters:
.  snes   - the SNES context

   Output Parameters:
.  sweeps  - the number of sweeps of GS to perform.

   Level: intermediate

.keywords: SNES, nonlinear, set, Gauss-Siedel

.seealso: SNESSetGS(), SNESGetGS(), SNESSetPC(), SNESGSSetSweeps()
@*/
PetscErrorCode SNESGSGetSweeps(SNES snes, PetscInt * sweeps)
{
  SNES_GS *gs = (SNES_GS*)snes->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  *sweeps = gs->sweeps;
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "SNESDefaultApplyGS"
PetscErrorCode SNESDefaultApplyGS(SNES snes,Vec X,Vec F,void *ctx)
{
  PetscFunctionBegin;
  /* see if there's a coloring on the DM */
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESReset_GS"
PetscErrorCode SNESReset_GS(SNES snes)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESDestroy_GS"
PetscErrorCode SNESDestroy_GS(SNES snes)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = SNESReset_GS(snes);CHKERRQ(ierr);
  ierr = PetscFree(snes->data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESSetUp_GS"
PetscErrorCode SNESSetUp_GS(SNES snes)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESSetFromOptions_GS"
PetscErrorCode SNESSetFromOptions_GS(SNES snes)
{
  SNES_GS        *gs = (SNES_GS*)snes->data;
  PetscErrorCode ierr;
  PetscInt       sweeps,max_its=PETSC_DEFAULT;
  PetscReal      rtol=PETSC_DEFAULT,atol=PETSC_DEFAULT,stol=PETSC_DEFAULT;
  PetscBool      flg,flg1,flg2,flg3;

  PetscFunctionBegin;
  ierr = PetscOptionsHead("SNES GS options");CHKERRQ(ierr);
  /* GS Options */
  ierr = PetscOptionsInt("-snes_gs_sweeps","Number of sweeps of GS to apply","SNESComputeGS",gs->sweeps,&sweeps,&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = SNESGSSetSweeps(snes,sweeps);CHKERRQ(ierr);
  }
  ierr = PetscOptionsReal("-snes_gs_atol","Number of sweeps of GS to apply","SNESComputeGS",gs->abstol,&atol,&flg);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-snes_gs_rtol","Number of sweeps of GS to apply","SNESComputeGS",gs->rtol,&rtol,&flg1);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-snes_gs_stol","Number of sweeps of GS to apply","SNESComputeGS",gs->stol,&stol,&flg2);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-snes_gs_max_it","Number of sweeps of GS to apply","SNESComputeGS",gs->max_its,&max_its,&flg3);CHKERRQ(ierr);
  if (flg || flg1 || flg2 || flg3) {
    ierr = SNESGSSetTolerances(snes,atol,rtol,stol,max_its);CHKERRQ(ierr);
  }
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESView_GS"
PetscErrorCode SNESView_GS(SNES snes, PetscViewer viewer)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESSolve_GS"
PetscErrorCode SNESSolve_GS(SNES snes)
{
  Vec              F;
  Vec              X;
  Vec              B;
  PetscInt         i;
  PetscReal        fnorm;
  PetscErrorCode   ierr;
  SNESNormSchedule normschedule;

  PetscFunctionBegin;
  X = snes->vec_sol;
  F = snes->vec_func;
  B = snes->vec_rhs;

  ierr         = PetscObjectSAWsTakeAccess((PetscObject)snes);CHKERRQ(ierr);
  snes->iter   = 0;
  snes->norm   = 0.;
  ierr         = PetscObjectSAWsGrantAccess((PetscObject)snes);CHKERRQ(ierr);
  snes->reason = SNES_CONVERGED_ITERATING;

  ierr = SNESGetNormSchedule(snes, &normschedule);CHKERRQ(ierr);
  if (normschedule == SNES_NORM_ALWAYS || normschedule == SNES_NORM_INITIAL_ONLY || normschedule == SNES_NORM_INITIAL_FINAL_ONLY) {
    /* compute the initial function and preconditioned update delX */
    if (!snes->vec_func_init_set) {
      ierr = SNESComputeFunction(snes,X,F);CHKERRQ(ierr);
      if (snes->domainerror) {
        snes->reason = SNES_DIVERGED_FUNCTION_DOMAIN;
        PetscFunctionReturn(0);
      }
    } else snes->vec_func_init_set = PETSC_FALSE;

    ierr = VecNorm(F, NORM_2, &fnorm);CHKERRQ(ierr); /* fnorm <- ||F||  */
    if (PetscIsInfOrNanReal(fnorm)) {
      snes->reason = SNES_DIVERGED_FNORM_NAN;
      PetscFunctionReturn(0);
    }
    ierr       = PetscObjectSAWsTakeAccess((PetscObject)snes);CHKERRQ(ierr);
    snes->iter = 0;
    snes->norm = fnorm;
    ierr       = PetscObjectSAWsGrantAccess((PetscObject)snes);CHKERRQ(ierr);
    ierr       = SNESLogConvergenceHistory(snes,snes->norm,0);CHKERRQ(ierr);
    ierr       = SNESMonitor(snes,0,snes->norm);CHKERRQ(ierr);

    /* test convergence */
    ierr = (*snes->ops->converged)(snes,0,0.0,0.0,fnorm,&snes->reason,snes->cnvP);CHKERRQ(ierr);
    if (snes->reason) PetscFunctionReturn(0);
  } else {
    ierr = PetscObjectSAWsGrantAccess((PetscObject)snes);CHKERRQ(ierr);
    ierr = SNESLogConvergenceHistory(snes,snes->norm,0);CHKERRQ(ierr);
    ierr = SNESMonitor(snes,0,snes->norm);CHKERRQ(ierr);
  }


  /* Call general purpose update function */
  if (snes->ops->update) {
    ierr = (*snes->ops->update)(snes, snes->iter);CHKERRQ(ierr);
  }

  for (i = 0; i < snes->max_its; i++) {
    ierr = SNESComputeGS(snes, B, X);CHKERRQ(ierr);
    /* only compute norms if requested or about to exit due to maximum iterations */
    if (normschedule == SNES_NORM_ALWAYS || ((i == snes->max_its - 1) && (normschedule == SNES_NORM_INITIAL_FINAL_ONLY || normschedule == SNES_NORM_FINAL_ONLY))) {
      ierr = SNESComputeFunction(snes,X,F);CHKERRQ(ierr);
      if (snes->domainerror) {
        snes->reason = SNES_DIVERGED_FUNCTION_DOMAIN;
        PetscFunctionReturn(0);
      }
      ierr = VecNorm(F, NORM_2, &fnorm);CHKERRQ(ierr); /* fnorm <- ||F||  */
      if (PetscIsInfOrNanReal(fnorm)) {
        snes->reason = SNES_DIVERGED_FNORM_NAN;
        PetscFunctionReturn(0);
      }
    }
    /* Monitor convergence */
    ierr       = PetscObjectSAWsTakeAccess((PetscObject)snes);CHKERRQ(ierr);
    snes->iter = i+1;
    snes->norm = fnorm;
    ierr       = PetscObjectSAWsGrantAccess((PetscObject)snes);CHKERRQ(ierr);
    ierr       = SNESLogConvergenceHistory(snes,snes->norm,0);CHKERRQ(ierr);
    ierr       = SNESMonitor(snes,snes->iter,snes->norm);CHKERRQ(ierr);
    /* Test for convergence */
    if (normschedule == SNES_NORM_ALWAYS) ierr = (*snes->ops->converged)(snes,snes->iter,0.0,0.0,fnorm,&snes->reason,snes->cnvP);CHKERRQ(ierr);
    if (snes->reason) PetscFunctionReturn(0);
    /* Call general purpose update function */
    if (snes->ops->update) {
      ierr = (*snes->ops->update)(snes, snes->iter);CHKERRQ(ierr);
    }
  }
  if (normschedule == SNES_NORM_ALWAYS) {
    if (i == snes->max_its) {
      ierr = PetscInfo1(snes,"Maximum number of iterations has been reached: %D\n",snes->max_its);CHKERRQ(ierr);
      if (!snes->reason) snes->reason = SNES_DIVERGED_MAX_IT;
    }
  } else if (!snes->reason) snes->reason = SNES_CONVERGED_ITS; /* GS is meant to be used as a preconditioner */
  PetscFunctionReturn(0);
}

/*MC
  SNESGS - Just calls the user-provided solution routine provided with SNESSetGS()

   Level: advanced

  Notes:
  the Gauss-Seidel smoother is inherited through composition.  If a solver has been created with SNESGetPC(), it will have
  its parent's Gauss-Seidel routine associated with it.

.seealso: SNESCreate(), SNES, SNESSetType(), SNESSetGS(), SNESType (for list of available types)
M*/

#undef __FUNCT__
#define __FUNCT__ "SNESCreate_GS"
PETSC_EXTERN PetscErrorCode SNESCreate_GS(SNES snes)
{
  SNES_GS        *gs;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  snes->ops->destroy        = SNESDestroy_GS;
  snes->ops->setup          = SNESSetUp_GS;
  snes->ops->setfromoptions = SNESSetFromOptions_GS;
  snes->ops->view           = SNESView_GS;
  snes->ops->solve          = SNESSolve_GS;
  snes->ops->reset          = SNESReset_GS;

  snes->usesksp = PETSC_FALSE;
  snes->usespc  = PETSC_FALSE;

  if (!snes->tolerancesset) {
    snes->max_its   = 10000;
    snes->max_funcs = 10000;
  }

  ierr = PetscNewLog(snes, SNES_GS, &gs);CHKERRQ(ierr);

  gs->sweeps  = 1;
  gs->rtol    = 1e-5;
  gs->abstol  = 1e-15;
  gs->stol    = 1e-12;
  gs->max_its = 50;

  snes->data = (void*) gs;
  PetscFunctionReturn(0);
}

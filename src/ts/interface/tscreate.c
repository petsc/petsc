
#include <petsc/private/tsimpl.h>      /*I "petscts.h"  I*/

const char *const TSConvergedReasons_Shifted[] = {
  "DIVERGED_STEP_REJECTED",
  "DIVERGED_NONLINEAR_SOLVE",
  "CONVERGED_ITERATING",
  "CONVERGED_TIME",
  "CONVERGED_ITS",
  "CONVERGED_USER",
  "CONVERGED_EVENT",
  "CONVERGED_PSEUDO_FATOL",
  "CONVERGED_PSEUDO_FATOL",
  "TSConvergedReason","TS_",0};
const char *const*TSConvergedReasons = TSConvergedReasons_Shifted + 2;

#undef  __FUNCT__
#define __FUNCT__ "TSCreate"
/*@C
  TSCreate - This function creates an empty timestepper. The problem type can then be set with TSSetProblemType() and the
       type of solver can then be set with TSSetType().

  Collective on MPI_Comm

  Input Parameter:
. comm - The communicator

  Output Parameter:
. ts   - The TS

  Level: beginner

.keywords: TS, create
.seealso: TSSetType(), TSSetUp(), TSDestroy(), TSSetProblemType()
@*/
PetscErrorCode  TSCreate(MPI_Comm comm, TS *ts)
{
  TS             t;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidPointer(ts,1);
  *ts = NULL;
  ierr = TSInitializePackage();CHKERRQ(ierr);

  ierr = PetscHeaderCreate(t, TS_CLASSID, "TS", "Time stepping", "TS", comm, TSDestroy, TSView);CHKERRQ(ierr);

  /* General TS description */
  t->problem_type      = TS_NONLINEAR;
  t->vec_sol           = NULL;
  t->numbermonitors    = 0;
  t->snes              = NULL;
  t->setupcalled       = 0;
  t->data              = NULL;
  t->user              = NULL;
  t->ptime             = 0.0;
  t->time_step         = 0.1;
  t->max_time          = 5.0;
  t->steprollback      = PETSC_FALSE;
  t->steps             = 0;
  t->max_steps         = 5000;
  t->ksp_its           = 0;
  t->snes_its          = 0;
  t->work              = NULL;
  t->nwork             = 0;
  t->max_snes_failures = 1;
  t->max_reject        = 10;
  t->errorifstepfailed = PETSC_TRUE;
  t->rhsjacobian.time  = -1e20;
  t->rhsjacobian.scale = 1.;
  t->ijacobian.shift   = 1.;
  t->equation_type     = TS_EQ_UNSPECIFIED;

  t->atol             = 1e-4;
  t->rtol             = 1e-4;
  t->cfltime          = PETSC_MAX_REAL;
  t->cfltime_local    = PETSC_MAX_REAL;
  t->exact_final_time = TS_EXACTFINALTIME_UNSPECIFIED;
  t->vec_costintegral = NULL;
  t->trajectory       = NULL;

  /* All methods that do not do adaptivity at all
   * should delete this object in their constructor */
  ierr = TSGetAdapt(t,&t->adapt);CHKERRQ(ierr);

  *ts = t;
  PetscFunctionReturn(0);
}

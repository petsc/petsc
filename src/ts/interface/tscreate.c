
#include <private/tsimpl.h>      /*I "petscts.h"  I*/

const char *const TSConvergedReasons_Shifted[] = {
  "DIVERGED_STEP_REJECTED",
  "DIVERGED_NONLINEAR_SOLVE",
  "CONVERGED_ITERATING",
  "CONVERGED_TIME",
  "CONVERGED_ITS",
  "TSConvergedReason","TS_",0};
const char *const*TSConvergedReasons = TSConvergedReasons_Shifted + 2;

#if 0
#undef __FUNCT__  
#define __FUNCT__ "TSPublish_Petsc"
static PetscErrorCode TSPublish_Petsc(PetscObject obj)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}
#endif

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
.seealso: TSSetType(), TSSetUp(), TSDestroy(), MeshCreate(), TSSetProblemType()
@*/
PetscErrorCode  TSCreate(MPI_Comm comm, TS *ts) {
  TS             t;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidPointer(ts,1);
  *ts = PETSC_NULL;
#ifndef PETSC_USE_DYNAMIC_LIBRARIES
  ierr = TSInitializePackage(PETSC_NULL);CHKERRQ(ierr);
#endif

  ierr = PetscHeaderCreate(t, _p_TS, struct _TSOps, TS_CLASSID, -1, "TS", comm, TSDestroy, TSView);CHKERRQ(ierr);
  ierr = PetscMemzero(t->ops, sizeof(struct _TSOps));CHKERRQ(ierr);

  /* General TS description */
  t->problem_type       = TS_NONLINEAR;
  t->vec_sol            = PETSC_NULL;
  t->numbermonitors     = 0;
  t->snes               = PETSC_NULL;
  t->funP               = PETSC_NULL;
  t->jacP               = PETSC_NULL;
  t->setupcalled        = 0;
  t->data               = PETSC_NULL;
  t->user               = PETSC_NULL;
  t->max_steps          = 5000;
  t->max_time           = 5.0;
  t->steps              = 0;
  t->linear_its         = 0;
  t->nonlinear_its      = 0;
  t->work               = PETSC_NULL;
  t->nwork              = 0;
  t->max_snes_failures  = 1;
  t->max_reject         = 10;
  t->errorifstepfailed  = PETSC_TRUE;
  t->rhsjacobian.time   = -1e20;

  ierr = TSSetInitialTimeStep(t,0.,0.1);CHKERRQ(ierr);

  *ts = t;
  PetscFunctionReturn(0);
}

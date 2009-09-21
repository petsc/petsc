#define PETSCTS_DLL

#include "private/tsimpl.h"      /*I "petscts.h"  I*/

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
PetscErrorCode PETSCTS_DLLEXPORT TSCreate(MPI_Comm comm, TS *ts) {
  TS             t;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidPointer(ts,1);
  *ts = PETSC_NULL;
#ifndef PETSC_USE_DYNAMIC_LIBRARIES
  ierr = TSInitializePackage(PETSC_NULL);CHKERRQ(ierr);
#endif

  ierr = PetscHeaderCreate(t, _p_TS, struct _TSOps, TS_COOKIE, -1, "TS", comm, TSDestroy, TSView);CHKERRQ(ierr);
  ierr = PetscMemzero(t->ops, sizeof(struct _TSOps));CHKERRQ(ierr);

  t->ops->prestep       = TSDefaultPreStep;
  t->ops->poststep      = TSDefaultPostStep;

  /* General TS description */
  t->problem_type       = TS_LINEAR;
  t->vec_sol            = PETSC_NULL;
  t->vec_sol_always     = PETSC_NULL;
  t->numbermonitors     = 0;
  t->ksp                = PETSC_NULL;
  t->A                  = PETSC_NULL;
  t->B                  = PETSC_NULL;
  t->Arhs               = PETSC_NULL;
  t->Alhs               = PETSC_NULL;
  t->matflg             = DIFFERENT_NONZERO_PATTERN;
  t->snes               = PETSC_NULL;
  t->funP               = PETSC_NULL;
  t->jacP               = PETSC_NULL;
  t->setupcalled        = 0;
  t->data               = PETSC_NULL;
  t->user               = PETSC_NULL;
  t->max_steps          = 5000;
  t->max_time           = 5.0;
  t->time_step          = .1;
  t->time_step_old      = t->time_step;
  t->initial_time_step  = t->time_step;
  t->steps              = 0;
  t->ptime              = 0.0;
  t->linear_its         = 0;
  t->nonlinear_its      = 0;
  t->work               = PETSC_NULL;
  t->nwork              = 0;

  *ts = t;
  PetscFunctionReturn(0);
}

/* Set A = 1/dt*Alhs - A, B = 1/dt*Blhs - B */
#undef __FUNCT__  
#define __FUNCT__ "TSScaleShiftMatrices"
PetscErrorCode TSScaleShiftMatrices(TS ts,Mat A,Mat B,MatStructure str)
{
  PetscTruth     flg;
  PetscErrorCode ierr;
  PetscScalar    mdt = 1.0/ts->time_step;

  PetscFunctionBegin;
  /* this function requires additional work! */
  ierr = PetscTypeCompare((PetscObject)A,MATMFFD,&flg);CHKERRQ(ierr);
  if (!flg) {
    ierr = MatScale(A,-1.0);CHKERRQ(ierr);
    if (ts->Alhs){
      ierr = MatAXPY(A,mdt,ts->Alhs,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
    } else {
      ierr = MatShift(A,mdt);CHKERRQ(ierr);
    }
  }
  if (B != A && str != SAME_PRECONDITIONER) {
    ierr = MatScale(B,-1.0);CHKERRQ(ierr);
    ierr = MatShift(B,mdt);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

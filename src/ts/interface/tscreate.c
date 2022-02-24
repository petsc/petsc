#include <petsc/private/tsimpl.h>      /*I "petscts.h"  I*/

const char *const TSConvergedReasons_Shifted[] = {
  "ADJOINT_DIVERGED_LINEAR_SOLVE",
  "FORWARD_DIVERGED_LINEAR_SOLVE",
  "DIVERGED_STEP_REJECTED",
  "DIVERGED_NONLINEAR_SOLVE",
  "CONVERGED_ITERATING",
  "CONVERGED_TIME",
  "CONVERGED_ITS",
  "CONVERGED_USER",
  "CONVERGED_EVENT",
  "CONVERGED_PSEUDO_FATOL",
  "CONVERGED_PSEUDO_FATOL",
  "TSConvergedReason","TS_",NULL};
const char *const*TSConvergedReasons = TSConvergedReasons_Shifted + 4;

/*@C
  TSCreate - This function creates an empty timestepper. The problem type can then be set with TSSetProblemType() and the
       type of solver can then be set with TSSetType().

  Collective

  Input Parameter:
. comm - The communicator

  Output Parameter:
. ts   - The TS

  Level: beginner

  Developer Notes:
    TS essentially always creates a SNES object even though explicit methods do not use it. This is
                    unfortunate and should be fixed at some point. The flag snes->usessnes indicates if the
                    particular method does use SNES and regulates if the information about the SNES is printed
                    in TSView(). TSSetFromOptions() does call SNESSetFromOptions() which can lead to users being confused
                    by help messages about meaningless SNES options.

.seealso: TSSetType(), TSSetUp(), TSDestroy(), TSSetProblemType()
@*/
PetscErrorCode  TSCreate(MPI_Comm comm, TS *ts)
{
  TS             t;

  PetscFunctionBegin;
  PetscValidPointer(ts,2);
  *ts = NULL;
  CHKERRQ(TSInitializePackage());

  CHKERRQ(PetscHeaderCreate(t, TS_CLASSID, "TS", "Time stepping", "TS", comm, TSDestroy, TSView));

  /* General TS description */
  t->problem_type      = TS_NONLINEAR;
  t->equation_type     = TS_EQ_UNSPECIFIED;

  t->ptime             = 0.0;
  t->time_step         = 0.1;
  t->max_time          = PETSC_MAX_REAL;
  t->exact_final_time  = TS_EXACTFINALTIME_UNSPECIFIED;
  t->steps             = 0;
  t->max_steps         = PETSC_MAX_INT;
  t->steprestart       = PETSC_TRUE;

  t->max_snes_failures = 1;
  t->max_reject        = 10;
  t->errorifstepfailed = PETSC_TRUE;

  t->rhsjacobian.time  = PETSC_MIN_REAL;
  t->rhsjacobian.scale = 1.0;
  t->ijacobian.shift   = 1.0;

  /* All methods that do adaptivity should specify
   * its preferred adapt type in their constructor */
  t->default_adapt_type = TSADAPTNONE;
  t->atol               = 1e-4;
  t->rtol               = 1e-4;
  t->cfltime            = PETSC_MAX_REAL;
  t->cfltime_local      = PETSC_MAX_REAL;

  t->num_rhs_splits     = 0;

  t->axpy_pattern       = UNKNOWN_NONZERO_PATTERN;
  *ts = t;
  PetscFunctionReturn(0);
}

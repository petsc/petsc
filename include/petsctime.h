/*
       Low cost access to a system time. This, in general, should not be included in user programs.
*/
#pragma once

#include <petscsys.h>

/* SUBMANSEC = Sys */

PETSC_EXTERN PetscErrorCode PetscGetCPUTime(PetscLogDouble *);

/* Global counters */
PETSC_EXTERN PetscLogDouble petsc_BaseTime;

/*@
   PetscTime - Returns the current time from some base time in the past in seconds.

   Not Collective

   Output Parameter:
.  v - time counter

   Usage:
.vb
     PetscLogDouble v;
     PetscTime(&v);
     .... perform some calculation ...
     printf("Time for operation %g\n",v);
.ve

   Level: developer

   Note:
   Since the PETSc libraries incorporate timing of phases and operations, we do not recommend ever using `PetscTime()`.
   The options database command  `-log_view` activates PETSc library timing.
   See `PetscLogStageRegister()`, `PetscLogEventRegister()`, `PetscLogEventBegin()`, `PetscLogEventEnd()` for how to register
   stages and events in application codes.

.seealso: `PetscTimeSubtract()`, `PetscTimeAdd()`, `PetscLogStageRegister()`, `PetscLogEventRegister()`, `PetscLogEventBegin()`, `PetscLogEventEnd()`
@*/
static inline PetscErrorCode PetscTime(PetscLogDouble *v)
{
  *v = MPI_Wtime();
  return PETSC_SUCCESS;
}

/*@
   PetscTimeSubtract - Subtracts the current time (in seconds) from the value `v`.

   Not Collective

   Input Parameter:
.  v - time counter

   Output Parameter:
.  v - time counter (`v` = `v` - current time)

   Level: developer

   Note:
   Since the PETSc libraries incorporate timing of phases and operations, we do not always recommend using `PetscTimeSubtract()`.
   The options database command  `-log_view` activates PETSc library timing.
   See `PetscLogStageRegister()`, `PetscLogEventRegister()`, `PetscLogEventBegin()`, `PetscLogEventEnd()` for how to register
   stages and events in application codes.

.seealso: `PetscTime()`, `PetscTimeAdd()`, `PetscLogStageRegister()`, `PetscLogEventRegister()`, `PetscLogEventBegin()`, `PetscLogEventEnd()`
@*/
static inline PetscErrorCode PetscTimeSubtract(PetscLogDouble *v)
{
  *v -= MPI_Wtime();
  return PETSC_SUCCESS;
}

/*@
   PetscTimeAdd - Adds the current time (in seconds) to the value `v`.

   Not Collective

   Input Parameter:
.  v - time counter

   Output Parameter:
.  v - time counter (`v` = `v` + current time)

   Level: developer

   Note:
   Since the PETSc libraries incorporate timing of phases and operations,  we do not ever recommend using `PetscTimeAdd()`.
   The options database command `-log_view` activates PETSc library timing.

.seealso: `PetscTime()`, `PetscTimeSubtract()`, `PetscLogStageRegister()`, `PetscLogEventRegister()`, `PetscLogEventBegin()`, `PetscLogEventEnd()`
@*/
static inline PetscErrorCode PetscTimeAdd(PetscLogDouble *v)
{
  *v += MPI_Wtime();
  return PETSC_SUCCESS;
}

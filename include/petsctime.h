/*
       Low cost access to system time. This, in general, should not
     be included in user programs.
*/

#if !defined(PETSCTIME_H)
#define PETSCTIME_H
#include <petscsys.h>

PETSC_EXTERN PetscErrorCode PetscGetCPUTime(PetscLogDouble*);

/* Global counters */
PETSC_EXTERN PetscLogDouble petsc_BaseTime;

/*MC
   PetscTime - Returns the current time of day in seconds.

   Synopsis:
    #include <petsctime.h>
    PetscErrorCode PetscTime(PetscLogDouble *v)

   Not Collective

   Output Parameter:
.  v - time counter


   Usage:
     PetscLogDouble v;
     PetscTime(&v);
     .... perform some calculation ...
     printf("Time for operation %g\n",v);

   Level: developer

   Notes:
   Since the PETSc libraries incorporate timing of phases and operations,
   we do not recommend ever using PetscTime()
   The options database command  -log_view activate
   PETSc library timing. See Users-Manual: ch_profiling for more details.

.seealso:  PetscTimeSubtract(), PetscTimeAdd(), PetscLogStageRegister(), PetscLogEventRegister(), PetscLogEventBegin(), PetscLogEventEnd()

M*/

/*MC
   PetscTimeSubtract - Subtracts the current time of day (in seconds) from
   the value v.

   Synopsis:
    #include <petsctime.h>
    PetscErrorCode PetscTimeSubtract(PetscLogDouble *v)

   Not Collective

   Input Parameter:
.  v - time counter

   Output Parameter:
.  v - time counter (v = v - current time)

   Level: developer

   Notes:
   Since the PETSc libraries incorporate timing of phases and operations,
   we do not every recommend using PetscTimeSubtract()
   The options database command  -log_view activates
   PETSc library timing.  See Users-Manual: ch_profiling for more details, also
   see PetscLogStageRegister(), PetscLogEventRegister(), PetscLogEventBegin(), PetscLogEventEnd() for how to register
   stages and events in application codes. 

.seealso:  PetscTime(), PetscTimeAdd(), PetscLogStageRegister(), PetscLogEventRegister(), PetscLogEventBegin(), PetscLogEventEnd()

M*/

/*MC
   PetscTimeAdd - Adds the current time of day (in seconds) to the value v.

   Synopsis:
    #include <petsctime.h>
    PetscErrorCode PetscTimeAdd(PetscLogDouble *v)

   Not Collective

   Input Parameter:
.  v - time counter

   Output Parameter:
.  v - time counter (v = v + current time)

   Level: developer

   Notes:
   Since the PETSc libraries incorporate timing of phases and operations,
   we do not ever recommend using PetscTimeAdd().
   The options database command -log_view activate
   PETSc library timing. See Users-Manual: ch_profiling for more details.

.seealso:  PetscTime(), PetscTimeSubtract(), PetscLogStageRegister(), PetscLogEventRegister(), PetscLogEventBegin(), PetscLogEventEnd()

M*/

PETSC_STATIC_INLINE PetscErrorCode PetscTime(PetscLogDouble *v)
{
  *v = MPI_Wtime();
  return 0;
}

PETSC_STATIC_INLINE PetscErrorCode PetscTimeSubtract(PetscLogDouble *v)
{
  *v -= MPI_Wtime();
  return 0;
}

PETSC_STATIC_INLINE PetscErrorCode PetscTimeAdd(PetscLogDouble *v)
{
  *v += MPI_Wtime();
  return 0;
}

#endif




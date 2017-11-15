/*
       Low cost access to system time. This, in general, should not
     be included in user programs.
*/

#if !defined(__PETSCTIME_H)
#define __PETSCTIME_H
#include <petscsys.h>

PETSC_EXTERN PetscErrorCode PetscGetCPUTime(PetscLogDouble*);

/* Global counters */
PETSC_EXTERN PetscLogDouble petsc_BaseTime;

/*MC
   PetscTime - Returns the current time of day in seconds.

   Synopsis:
    #include "petsctime.h"
   PetscTime(PetscLogDouble *v)

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
   we do not recomment every using PetscTime()
   The options database command  -log_summary activate
   PETSc library timing. See the <A href="../../docs/manual.pdf">Users Manual</A> for more details.

.seealso:  PetscTimeSubtract(), PetscTimeAdd(), PetscLogStageRegister(), PetscLogEventRegister(), PetscLogEventBegin(), PetscLogEventEnd()

.keywords:  Petsc, time
M*/

/*MC
   PetscTimeSubtract - Subtracts the current time of day (in seconds) from
   the value v.

   Synopsis:
    #include "petsctime.h"
   PetscTimeSubtract(&PetscLogDouble *v)

   Not Collective

   Input Parameter:
.  v - time counter

   Output Parameter:
.  v - time counter (v = v - current time)

   Level: developer

   Notes:
   Since the PETSc libraries incorporate timing of phases and operations,
   we do not every recommend using PetscTimeSubtract()
   The options database command  -log_summary activates
   PETSc library timing.  See the <A href="../../docs/manual.pdf">Users Manual</A> for more details, also
   see PetscLogStageRegister(), PetscLogEventRegister(), PetscLogEventBegin(), PetscLogEventEnd() for how to register
   stages and events in application codes. 

.seealso:  PetscTime(), PetscTimeAdd(), PetscLogStageRegister(), PetscLogEventRegister(), PetscLogEventBegin(), PetscLogEventEnd()

.keywords:  Petsc, time, subtract
M*/

/*MC
   PetscTimeAdd - Adds the current time of day (in seconds) to the value v.

   Synopsis:
    #include "petsctime.h"
   PetscTimeAdd(PetscLogDouble *v)

   Not Collective

   Input Parameter:
.  v - time counter

   Output Parameter:
.  v - time counter (v = v + current time)

   Level: developer

   Notes:
   Since the PETSc libraries incorporate timing of phases and operations,
   we do not ever recommend using PetscTimeAdd().
   The options database command -log_summary activate
   PETSc library timing. See the <A href="../../docs/manual.pdf">Users Manual</A> for more details.

.seealso:  PetscTime(), PetscTimeSubtract(), PetscLogStageRegister(), PetscLogEventRegister(), PetscLogEventBegin(), PetscLogEventEnd()

.keywords:  Petsc, time, add
M*/

/* ------------------------------------------------------------------
    Some machines have very fast MPI_Wtime()
*/
#if (defined(PETSC_HAVE_FAST_MPI_WTIME) && !defined(__MPIUNI_H))
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

/* ------------------------------------------------------------------
   IBM Power and PowerPC machines have a fast clock read_real_time()
*/
#elif defined(PETSC_USE_READ_REAL_TIME)
PETSC_EXTERN PetscLogDouble PetscReadRealTime(void);

PETSC_STATIC_INLINE PetscErrorCode PetscTime(PetscLogDouble *v)
{
  *v = PetscReadRealTime();
  return 0;
}

PETSC_STATIC_INLINE PetscErrorCode PetscTimeSubtract(PetscLogDouble *v)
{
  *v -= PetscReadRealTime();
  return 0;
}

PETSC_STATIC_INLINE PetscErrorCode PetscTimeAdd(PetscLogDouble *v)
{
  *v += PetscReadRealTime();
  return 0;
}

/* ------------------------------------------------------------------
   Microsoft Windows has its own time routines
*/
#elif defined (PETSC_USE_MICROSOFT_TIME)
#include <time.h>
PETSC_EXTERN PetscLogDouble PetscMicrosoftTime(void);

PETSC_STATIC_INLINE PetscErrorCode PetscTime(PetscLogDouble *v)
{
  *v = PetscMicrosoftTime();
  return 0;
}

PETSC_STATIC_INLINE PetscErrorCode PetscTimeSubtract(PetscLogDouble *v)
{
  *v -= PetscMicrosoftTime();
  return 0;
}

PETSC_STATIC_INLINE PetscErrorCode PetscTimeAdd(PetscLogDouble *v)
{
  *v += PetscMicrosoftTime();
  return 0;
}

/* ------------------------------------------------------------------
    The usual Unix time routines.
*/
#else

#if defined(PETSC_HAVE_SYS_TIME_H)
#include <sys/time.h>
#endif

#if defined(PETSC_NEEDS_GETTIMEOFDAY_PROTO)
PETSC_EXTERN int gettimeofday(struct timeval *,struct timezone *);
#endif

PETSC_STATIC_INLINE PetscErrorCode PetscTime(PetscLogDouble *v)
{
  static struct timeval _tp;
  gettimeofday(&_tp,(struct timezone *)0);
  *v = ((PetscLogDouble)_tp.tv_sec)+(1.0e-6)*(_tp.tv_usec);
  return 0;
}

PETSC_STATIC_INLINE PetscErrorCode PetscTimeSubtract(PetscLogDouble *v)
{
  static struct timeval _tp;
  gettimeofday(&_tp,(struct timezone *)0);
  *v -= ((PetscLogDouble)_tp.tv_sec)+(1.0e-6)*(_tp.tv_usec);
  return 0;
}

PETSC_STATIC_INLINE PetscErrorCode PetscTimeAdd(PetscLogDouble *v)
{
  static struct timeval _tp;
  gettimeofday(&_tp,(struct timezone *)0);
  *v += ((PetscLogDouble)_tp.tv_sec)+(1.0e-6)*(_tp.tv_usec);
  return 0;
}

#endif

#endif




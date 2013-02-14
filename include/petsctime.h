/*
       Low cost access to system time. This, in general, should not
     be included in user programs.
*/

#if !defined(__PETSCTIME_H)
#define __PETSCTIME_H

#include <petscsys.h>
#if defined(PETSC_HAVE_SYS_TIME_H)
#include <sys/time.h>
#endif
#if defined(PETSC_NEEDS_GETTIMEOFDAY_PROTO)
EXTERN_C_BEGIN
extern int gettimeofday(struct timeval *,struct timezone *);
EXTERN_C_END
#endif

/* Global counters */
PETSC_EXTERN PetscLogDouble petsc_BaseTime;

/*MC
   PetscTime - Returns the current time of day in seconds.

   Synopsis:
    #include "petsctime.h"
   PetscTime(PetscLogDouble v)

   Not Collective

   Output Parameter:
.  v - time counter


   Usage:
     PetscLogDouble v;
     PetscTime(v);
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
   PetscTimeSubtract(PetscLogDouble v)

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
   PetscTimeAdd(PetscLogDouble v)

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
#define PetscTime(v)         (v)=MPI_Wtime();

#define PetscTimeSubtract(v) (v)-=MPI_Wtime();

#define PetscTimeAdd(v)      (v)+=MPI_Wtime();

/* ------------------------------------------------------------------
   IBM Power and PowerPC machines have a fast clock read_real_time()
*/
#elif defined(PETSC_USE_READ_REAL_TIME)
PETSC_EXTERN PetscLogDouble PetscReadRealTime(void);
#define PetscTime(v)         (v)=PetscReadRealTime();

#define PetscTimeSubtract(v) (v)-=PetscReadRealTime();

#define PetscTimeAdd(v)      (v)+=PetscReadRealTime();

/* ------------------------------------------------------------------
   Microsoft Windows has its own time routines
*/
#elif defined (PETSC_USE_MICROSOFT_TIME)
#include <time.h>
EXTERN_C_BEGIN
PETSC_EXTERN PetscLogDouble PetscMicrosoftTime(void);
EXTERN_C_END
#define PetscTime(v)         (v)=PetscMicrosoftTime();

#define PetscTimeSubtract(v) (v)-=PetscMicrosoftTime();

#define PetscTimeAdd(v)      (v)+=PetscMicrosoftTime();

/* ------------------------------------------------------------------
    The usual Unix time routines.
*/
#else
#define PetscTime(v)         do {                                       \
    static struct timeval _tp;                                          \
    gettimeofday(&_tp,(struct timezone *)0);                            \
    (v)=((PetscLogDouble)_tp.tv_sec)+(1.0e-6)*(_tp.tv_usec);            \
  } while (0)

#define PetscTimeSubtract(v) do {                                       \
    static struct timeval _tp;                                          \
    gettimeofday(&_tp,(struct timezone *)0);                            \
    (v)-=((PetscLogDouble)_tp.tv_sec)+(1.0e-6)*(_tp.tv_usec);           \
  } while (0)

#define PetscTimeAdd(v) do {                                            \
    static struct timeval _tp;                                          \
    gettimeofday(&_tp,(struct timezone *)0);                            \
    (v)+=((PetscLogDouble)_tp.tv_sec)+(1.0e-6)*(_tp.tv_usec);           \
  } while (0)
#endif

#endif





